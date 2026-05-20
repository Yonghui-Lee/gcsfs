#include <Python.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

/* * Global references to Python functions.
 * We look these up once at init time to save overhead.
 */
static PyObject *pModule = NULL;
static PyObject *pFuncInit = NULL;
static PyObject *pFuncOpen = NULL;
static PyObject *pFuncClose = NULL;
static PyObject *pFuncQueue = NULL;
static PyObject *pFuncGetEvents = NULL;

/*
 * Thread-local data to store events retrieved from Python
 * before passing them one-by-one to FIO.
 */
struct py_thread_data {
    PyObject *reaped_events; // List of completed events
    int reaped_index;        // Current index in the list
};

/*
 * FIO specific options for our engine
 */
struct py_options {
    void *pad;
    unsigned int iodepth; 
    unsigned int flush_every_write;
};

static struct fio_option options[] = {
    {
        .name = "flush_every_write",
        .lname = "flush_every_write",
        .type = FIO_OPT_BOOL,
        .off1 = offsetof(struct py_options, flush_every_write),
        .def = "0",
        .help = "If true, flushes the writer after every append",
        .category = FIO_OPT_C_ENGINE,
        .group = FIO_OPT_G_INVALID,
    },
    {
        .name = NULL,
    },
};

/*
 * Initialize the Python Interpreter and import our module.
 */
static int py_storage_init(struct thread_data *td) {
    // 1. Initialize Python (only once per process)
    if (!Py_IsInitialized()) {
        Py_Initialize();
        
        // Add current directory to sys.path so we can find gcsfs_adapter.py
        PyObject *sysPath = PySys_GetObject("path");
        PyObject *cwd = PyUnicode_FromString(".");
        PyList_Append(sysPath, cwd);
        Py_DECREF(cwd);

        // Import the module
        PyObject *pName = PyUnicode_FromString("gcsfs_adapter");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule == NULL) {
            PyErr_Print();
            fprintf(stderr, "Failed to load module 'gcsfs_adapter'\n");
            return 1;
        }

        // Look up functions
        pFuncInit = PyObject_GetAttrString(pModule, "py_init");
        pFuncOpen = PyObject_GetAttrString(pModule, "py_open");
        pFuncClose = PyObject_GetAttrString(pModule, "py_close");
        pFuncQueue = PyObject_GetAttrString(pModule, "py_queue");
        pFuncGetEvents = PyObject_GetAttrString(pModule, "py_get_events");

        if (!pFuncInit || !pFuncOpen || !pFuncClose || !pFuncQueue || !pFuncGetEvents) {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Failed to find required Python functions.\n");
            return 1;
        }
    }

    // 2. Initialize the Python-side logic (Global Loop & Client)
    // Acquire GIL for calling Python
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    PyObject *args = PyTuple_Pack(1, PyLong_FromLong(td->o.iodepth));
    PyObject *result = PyObject_CallObject(pFuncInit, args);
    Py_DECREF(args);
    
    int success = 0;
    if (result == NULL) {
        PyErr_Print();
        success = 1;
    } else {
        long ret = PyLong_AsLong(result);
        Py_DECREF(result);
        success = (ret == 0) ? 0 : 1;
    }

    // 3. Setup thread-local data
    struct py_thread_data *ptd = calloc(1, sizeof(struct py_thread_data));
    td->io_ops_data = ptd;

    PyGILState_Release(gstate);
    return success;
}

static void py_storage_cleanup(struct thread_data *td) {
    struct py_thread_data *ptd = td->io_ops_data;
    if (ptd) {
        if (ptd->reaped_events) {
            PyGILState_STATE gstate = PyGILState_Ensure();
            Py_DECREF(ptd->reaped_events);
            PyGILState_Release(gstate);
        }
        free(ptd);
    }
}

static int py_storage_open(struct thread_data *td, struct fio_file *f) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    struct py_options *o = td->eo;
    int is_write = (td->o.td_ddir == TD_DDIR_WRITE);
    
    PyObject *args = PyTuple_Pack(4, 
        PyUnicode_FromString(f->file_name),
        PyBool_FromLong(is_write),
        PyBool_FromLong(o->flush_every_write),
        PyLong_FromLongLong((long long)f->real_file_size)
    );
    
    PyObject *result = PyObject_CallObject(pFuncOpen, args);
    Py_DECREF(args);

    if (result == NULL) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 1;
    }

    long handle = PyLong_AsLong(result);
    Py_DECREF(result);

    // Store the python handle ID in the fio file structure
    f->engine_data = (void *)(uintptr_t)handle;

    PyGILState_Release(gstate);
    return (handle == 0) ? 1 : 0;
}

static int py_storage_close(struct thread_data *td, struct fio_file *f) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    long handle = (long)(uintptr_t)f->engine_data;
    PyObject *args = PyTuple_Pack(1, PyLong_FromLong(handle));
    PyObject *result = PyObject_CallObject(pFuncClose, args);
    Py_DECREF(args);
    
    if (result) Py_DECREF(result);
    else PyErr_Print();

    PyGILState_Release(gstate);
    return 0;
}

static enum fio_q_status py_storage_queue(struct thread_data *td, struct io_u *io_u) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    long handle = (long)(uintptr_t)io_u->file->engine_data;
    int is_write = (io_u->ddir == DDIR_WRITE);
    
    // Zero-Copy Magic: Create a MemoryView directly on the FIO C buffer.
    // Python can write directly into this (for reads) or read from it (for writes).
    PyObject *py_buf = PyMemoryView_FromMemory(
        (char *)io_u->xfer_buf, 
        io_u->xfer_buflen, 
        is_write ? PyBUF_READ : PyBUF_WRITE
    );

    // We pass the io_u pointer address as the 'tag' so we can identify it later
    PyObject *args = PyTuple_Pack(5,
        PyLong_FromLong(handle),
        PyLong_FromVoidPtr(io_u),
        PyLong_FromLongLong(io_u->offset),
        py_buf,
        PyBool_FromLong(is_write)
    );

    PyObject *result = PyObject_CallObject(pFuncQueue, args);
    Py_DECREF(args);
    Py_DECREF(py_buf); 

    enum fio_q_status ret = FIO_Q_COMPLETED; 
    if (result == NULL) {
        PyErr_Print();
        io_u->error = EIO;
    } else {
        long val = PyLong_AsLong(result);
        Py_DECREF(result);
        if (val == 1) ret = FIO_Q_QUEUED;
        else if (val == 0) ret = FIO_Q_COMPLETED;
        else {
            io_u->error = EIO;
            ret = FIO_Q_COMPLETED;
        }
    }

    PyGILState_Release(gstate);
    return ret;
}

/*
 * Waits for events from Python and stores them in thread-local storage.
 */
static int py_storage_getevents(struct thread_data *td, unsigned int min, unsigned int max, const struct timespec *t) {
    struct py_thread_data *ptd = td->io_ops_data;
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Clean up previous batch if it exists
    if (ptd->reaped_events) {
        Py_DECREF(ptd->reaped_events);
        ptd->reaped_events = NULL;
    }
    ptd->reaped_index = 0;

    PyObject *args = PyTuple_Pack(1, PyLong_FromLong(min));
    PyObject *result_list = PyObject_CallObject(pFuncGetEvents, args);
    Py_DECREF(args);

    if (result_list == NULL) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return -1;
    }

    if (!PyList_Check(result_list)) {
        fprintf(stderr, "py_get_events did not return a list\n");
        Py_DECREF(result_list);
        PyGILState_Release(gstate);
        return -1;
    }

    int count = (int)PyList_Size(result_list);
    ptd->reaped_events = result_list; // Store list for event() calls

    PyGILState_Release(gstate);
    return count;
}

/*
 * Returns the actual IO unit for the next completed event.
 */
static struct io_u *py_storage_event(struct thread_data *td, int event) {
    struct py_thread_data *ptd = td->io_ops_data;
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *item = PyList_GetItem(ptd->reaped_events, ptd->reaped_index); // Borrowed ref
    ptd->reaped_index++;

    // Item is tuple (tag_ptr, errno)
    PyObject *pTag = PyTuple_GetItem(item, 0);
    PyObject *pErr = PyTuple_GetItem(item, 1);

    struct io_u *io_u = (struct io_u *)PyLong_AsVoidPtr(pTag);
    int err = (int)PyLong_AsLong(pErr);

    if (err != 0) {
        io_u->error = EIO; 
    } else {
        io_u->error = 0;
    }
    
    PyGILState_Release(gstate);
    return io_u;
}

struct ioengine_ops ioengine = {
    .name = "gcsfs",
    .version = FIO_IOOPS_VERSION,
    .init = py_storage_init,
    .cleanup = py_storage_cleanup,
    .open_file = py_storage_open,
    .close_file = py_storage_close,
    .queue = py_storage_queue,
    .getevents = py_storage_getevents,
    .event = py_storage_event,
    .flags = FIO_DISKLESSIO | FIO_NOEXTEND | FIO_NODISKUTIL,
    .options = options,
    .option_struct_size = sizeof(struct py_options),
};
