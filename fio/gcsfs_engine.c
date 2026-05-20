#include <Python.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <poll.h>
#include <unistd.h>
#include <sys/eventfd.h>

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

/* Global references to Python functions.
 * We look these up once at init time to save overhead.
 */
static PyObject *pModule = NULL;
static PyObject *pFuncInit = NULL;
static PyObject *pFuncOpen = NULL;
static PyObject *pFuncClose = NULL;
static PyObject *pFuncQueue = NULL;

#define RING_BUFFER_SIZE 1024

struct reaped_event {
    struct io_u *io_u;
    int error;
};

struct py_spsc_ring {
    struct reaped_event events[RING_BUFFER_SIZE];
    _Atomic unsigned int head;
    _Atomic unsigned int tail;
};

/*
 * Thread-local data to store events reaped via the lockless SPSC ring buffer.
 */
struct py_thread_data {
    struct py_spsc_ring ring;
    int event_fd;
    struct reaped_event events[RING_BUFFER_SIZE];
    int events_count;
    int events_index;
};

static struct py_thread_data *global_ptd = NULL;
static PyThreadState *main_tstate = NULL;

static inline int ring_push(struct py_spsc_ring *ring, struct io_u *io_u, int error) {
    unsigned int t = atomic_load_explicit(&ring->tail, memory_order_relaxed);
    unsigned int h = atomic_load_explicit(&ring->head, memory_order_acquire);
    
    if ((t - h) >= RING_BUFFER_SIZE) {
        return -1; // Queue full
    }
    
    unsigned int idx = t & (RING_BUFFER_SIZE - 1);
    ring->events[idx].io_u = io_u;
    ring->events[idx].error = error;
    
    atomic_store_explicit(&ring->tail, t + 1, memory_order_release);
    return 0;
}

static inline int ring_pop(struct py_spsc_ring *ring, struct reaped_event *out_event) {
    unsigned int h = atomic_load_explicit(&ring->head, memory_order_relaxed);
    unsigned int t = atomic_load_explicit(&ring->tail, memory_order_acquire);
    
    if (h == t) {
        return -1; // Queue empty
    }
    
    unsigned int idx = h & (RING_BUFFER_SIZE - 1);
    *out_event = ring->events[idx];
    
    atomic_store_explicit(&ring->head, h + 1, memory_order_release);
    return 0;
}

void c_complete_trampoline(void *io_u_ptr, int err) {
    struct io_u *io_u = (struct io_u *)io_u_ptr;
    
    if (!global_ptd) {
        fprintf(stderr, "c_complete_trampoline called before global_ptd was set!\n");
        return;
    }
    
    if (ring_push(&global_ptd->ring, io_u, err) < 0) {
        fprintf(stderr, "SPSC Ring Buffer overflow in c_complete_trampoline!\n");
        return;
    }
    
    eventfd_t val = 1;
    if (eventfd_write(global_ptd->event_fd, val) < 0) {
        perror("eventfd_write failed in trampoline");
    }
}

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
        // Force libpython symbols to be globally visible before initializing
        void *libpython = dlopen(PY_SONAME, RTLD_GLOBAL | RTLD_NOW);
        if (!libpython) {
            fprintf(stderr, "Warning: Failed to dlopen %s globally: %s\n", PY_SONAME, dlerror());
        }

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

        if (!pFuncInit || !pFuncOpen || !pFuncClose || !pFuncQueue) {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Failed to find required Python functions.\n");
            return 1;
        }

        main_tstate = PyEval_SaveThread();
    }

    // 2. Setup thread-local data
    struct py_thread_data *ptd = calloc(1, sizeof(struct py_thread_data));
    td->io_ops_data = ptd;

    ptd->event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (ptd->event_fd < 0) {
        perror("Failed to create eventfd");
        free(ptd);
        td->io_ops_data = NULL;
        return 1;
    }

    global_ptd = ptd;

    // 3. Initialize the Python-side logic (Global Loop & Client)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *args = PyTuple_Pack(2, 
        PyLong_FromLong(td->o.iodepth), 
        PyLong_FromVoidPtr((void *)c_complete_trampoline)
    );
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

    PyGILState_Release(gstate);
    return success;
}

static void py_storage_cleanup(struct thread_data *td) {
    struct py_thread_data *ptd = td->io_ops_data;
    if (ptd) {
        if (ptd->event_fd >= 0) {
            close(ptd->event_fd);
        }
        if (global_ptd == ptd) {
            global_ptd = NULL;
        }
        free(ptd);
        td->io_ops_data = NULL;
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
 * Waits for events from the lockless SPSC ring buffer and blocks on eventfd if empty.
 * This is completely GIL-free!
 */
static int py_storage_getevents(struct thread_data *td, unsigned int min, unsigned int max, const struct timespec *t) {
    struct py_thread_data *ptd = td->io_ops_data;
    
    ptd->events_count = 0;
    ptd->events_index = 0;
    
    int timeout_ms = -1;
    if (t) {
        timeout_ms = t->tv_sec * 1000 + t->tv_nsec / 1000000;
    }
    
    unsigned int reaped = 0;
    
    while (reaped < max) {
        struct reaped_event ev;
        if (ring_pop(&ptd->ring, &ev) == 0) {
            ptd->events[reaped] = ev;
            reaped++;
            continue;
        }
        
        if (reaped >= min) {
            break;
        }
        
        struct pollfd pfd;
        pfd.fd = ptd->event_fd;
        pfd.events = POLLIN;
        
        if (timeout_ms == 0) {
            break;
        }
        
        int poll_ret = poll(&pfd, 1, timeout_ms);
        if (poll_ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("poll failed in py_storage_getevents");
            return -1;
        } else if (poll_ret == 0) {
            break; // Timeout
        } else {
            eventfd_t val;
            if (eventfd_read(ptd->event_fd, &val) < 0) {
                if (errno != EAGAIN) {
                    perror("eventfd_read failed");
                }
            }
        }
    }
    
    ptd->events_count = reaped;
    return reaped;
}

/*
 * Returns the actual IO unit for the next completed event.
 * This is completely GIL-free!
 */
static struct io_u *py_storage_event(struct thread_data *td, int event) {
    struct py_thread_data *ptd = td->io_ops_data;
    
    if (event >= ptd->events_count) {
        return NULL;
    }
    
    struct io_u *io_u = ptd->events[event].io_u;
    int err = ptd->events[event].error;
    
    if (err != 0) {
        io_u->error = EIO;
    } else {
        io_u->error = 0;
    }
    
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
