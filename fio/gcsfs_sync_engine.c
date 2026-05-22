// fio/gcsfs_sync_engine.c

#include <Python.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

/* Global references to Python objects and functions.
 * We resolve these once at init time to keep operations lightweight.
 */
static PyObject *pModule = NULL;
static PyObject *pFuncInit = NULL;
static PyObject *pFuncOpen = NULL;
static PyObject *pFuncClose = NULL;
static PyObject *pFuncRead = NULL;
static PyObject *pFuncWrite = NULL;

static PyThreadState *main_tstate = NULL;
static pthread_mutex_t import_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * FIO custom options specifically for the synchronous engine
 */
struct py_sync_options {
    void *pad;
    unsigned int block_size;
    unsigned int use_prefetch;
    unsigned int concurrency;
};

static struct fio_option options[] = {
    {
        .name = "block_size",
        .lname = "block_size",
        .type = FIO_OPT_INT,
        .off1 = offsetof(struct py_sync_options, block_size),
        .def = "16777216", // 16MB default
        .help = "Read-ahead buffer block size in bytes",
        .category = FIO_OPT_C_ENGINE,
        .group = FIO_OPT_G_INVALID,
    },
    {
        .name = "use_prefetch",
        .lname = "use_prefetch",
        .type = FIO_OPT_BOOL,
        .off1 = offsetof(struct py_sync_options, use_prefetch),
        .def = "1",
        .help = "Enable adaptive background prefetcher (requires read mode)",
        .category = FIO_OPT_C_ENGINE,
        .group = FIO_OPT_G_INVALID,
    },
    {
        .name = "concurrency",
        .lname = "concurrency",
        .type = FIO_OPT_INT,
        .off1 = offsetof(struct py_sync_options, concurrency),
        .def = "4",
        .help = "Number of concurrent requests to fetch the data",
        .category = FIO_OPT_C_ENGINE,
        .group = FIO_OPT_G_INVALID,
    },
    {
        .name = NULL,
    },
};

/*
 * Initialize the CPython interpreter and load our synchronous python module.
 * Executed once per FIO job thread.
 */
static int py_sync_storage_init(struct thread_data *td) {
    struct py_sync_options *o = td->eo;

    pthread_mutex_lock(&import_mutex);

    // 1. Initialize Python Runtime (only once per process)
    if (!Py_IsInitialized()) {
        // Set the default GCSFS concurrency environment variable before importing modules.
        // This ensures that zb_hns_utils.py picks up the value during its initial import.
        char buf[32];
        snprintf(buf, sizeof(buf), "%u", o->concurrency);
        setenv("DEFAULT_GCSFS_CONCURRENCY", buf, 1);

        // Force libpython symbols to be globally visible (crucial for loading native extension modules)
        void *libpython = dlopen(PY_SONAME, RTLD_GLOBAL | RTLD_NOW);
        if (!libpython) {
            fprintf(stderr, "Warning: Failed to dlopen %s globally: %s\n", PY_SONAME, dlerror());
        }

        Py_Initialize();

        // Append current directory to sys.path to resolve local python modules
        PyObject *sysPath = PySys_GetObject("path");
        PyObject *cwd = PyUnicode_FromString(".");
        PyList_Append(sysPath, cwd);
        Py_DECREF(cwd);

        // Load the Python synchronous adapter module
        PyObject *pName = PyUnicode_FromString("gcsfs_sync_adapter");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule == NULL) {
            PyErr_Print();
            fprintf(stderr, "Failed to load Python module 'gcsfs_sync_adapter'\n");
            pthread_mutex_unlock(&import_mutex);
            return 1;
        }

        // Bind reference callbacks
        pFuncInit = PyObject_GetAttrString(pModule, "py_sync_init");
        pFuncOpen = PyObject_GetAttrString(pModule, "py_sync_open");
        pFuncClose = PyObject_GetAttrString(pModule, "py_sync_close");
        pFuncRead = PyObject_GetAttrString(pModule, "py_sync_read");
        pFuncWrite = PyObject_GetAttrString(pModule, "py_sync_write");

        if (!pFuncInit || !pFuncOpen || !pFuncClose || !pFuncRead || !pFuncWrite) {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Failed to resolve required python callbacks in gcsfs_sync_adapter.\n");
            pthread_mutex_unlock(&import_mutex);
            return 1;
        }

        // Enable multi-threading: Save thread state and release global interpreter lock (GIL)
        main_tstate = PyEval_SaveThread();
    }

    pthread_mutex_unlock(&import_mutex);

    // 2. Run Python-side setup
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *result = PyObject_CallObject(pFuncInit, NULL);
    int success = 0;
    if (result == NULL) {
        PyErr_Print();
        success = 1;
    } else {
        long ret = PyLong_AsLong(result);
        Py_DECREF(result);
        success = (ret == 0) ? 0 : 1;
    }

    PyGILState_Release(gstate);
    return success;
}

static void py_sync_storage_cleanup(struct thread_data *td) {
    // No-op. Option strings are automatically freed by FIO.
}

/*
 * Open the target file. Invoked during FIO start or preparation.
 */
static int py_sync_storage_open(struct thread_data *td, struct fio_file *f) {
    struct py_sync_options *o = td->eo;
    int is_write = (td->o.td_ddir == TD_DDIR_WRITE);

    PyGILState_STATE gstate = PyGILState_Ensure();

    // Invoke open callback passing open context parameters
    PyObject *args = PyTuple_Pack(5,
        PyUnicode_FromString(f->file_name),
        PyBool_FromLong(is_write),
        PyLong_FromLong(o->block_size),
        PyBool_FromLong(o->use_prefetch),
        PyLong_FromLong(o->concurrency)
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

    // Keep handle registration in FIO file context
    f->engine_data = (void *)(uintptr_t)handle;

    PyGILState_Release(gstate);
    return (handle == 0) ? 1 : 0;
}

/*
 * Close the target file.
 */
static int py_sync_storage_close(struct thread_data *td, struct fio_file *f) {
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

/*
 * Synchronous queue method. Excutes block I/O completely synchronously.
 */
static enum fio_q_status py_sync_storage_queue(struct thread_data *td, struct io_u *io_u) {
    long handle = (long)(uintptr_t)io_u->file->engine_data;
    int is_write = (io_u->ddir == DDIR_WRITE);

    PyGILState_STATE gstate = PyGILState_Ensure();

    // Create a zero-copy memoryview wrapper referencing the FIO C transfer buffer
    PyObject *memview = PyMemoryView_FromMemory(
        (char *)io_u->xfer_buf,
        io_u->xfer_buflen,
        is_write ? PyBUF_READ : PyBUF_WRITE
    );

    if (!memview) {
        PyErr_Print();
        PyGILState_Release(gstate);
        io_u->error = EIO;
        return FIO_Q_COMPLETED;
    }

    PyObject *args = PyTuple_Pack(3,
        PyLong_FromLong(handle),
        PyLong_FromLongLong(io_u->offset),
        memview
    );

    PyObject *result;
    if (is_write) {
        result = PyObject_CallObject(pFuncWrite, args);
    } else {
        result = PyObject_CallObject(pFuncRead, args);
    }

    Py_DECREF(args);
    Py_DECREF(memview);

    long ret_bytes = -1;
    if (result == NULL) {
        PyErr_Print();
    } else {
        ret_bytes = PyLong_AsLong(result);
        Py_DECREF(result);
    }

    PyGILState_Release(gstate);

    // Set outcomes
    if (ret_bytes < 0) {
        io_u->error = EIO;
    } else {
        io_u->resid = io_u->xfer_buflen - ret_bytes;
        io_u->error = 0;
    }

    // Return FIO_Q_COMPLETED to signal immediate inline I/O resolution
    return FIO_Q_COMPLETED;
}

/*
 * Exported ioengine operations descriptor for dynamic linking
 */
struct ioengine_ops ioengine = {
    .name = "gcsfs_sync",
    .version = FIO_IOOPS_VERSION,
    .init = py_sync_storage_init,
    .cleanup = py_sync_storage_cleanup,
    .open_file = py_sync_storage_open,
    .close_file = py_sync_storage_close,
    .queue = py_sync_storage_queue,
    .flags = FIO_DISKLESSIO | FIO_NOEXTEND | FIO_NODISKUTIL | FIO_SYNCIO,
    .options = options,
    .option_struct_size = sizeof(struct py_sync_options),
};
