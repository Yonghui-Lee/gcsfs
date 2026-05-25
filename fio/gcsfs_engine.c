#include <Python.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <poll.h>
#include <sched.h>
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
static PyObject *pFuncGetSize = NULL;

#define RING_BUFFER_SIZE 1024

struct reaped_event {
    struct io_u *io_u;
    int error;
    unsigned long long bytes_done;
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

struct gcsfs_io_u_data {
    void *memview;
    char *cached_buf;
    unsigned long long cached_len;
};

static PyThreadState *main_tstate = NULL;

static inline int ring_push(struct py_spsc_ring *ring, struct io_u *io_u, int error, unsigned long long bytes_done) {
    unsigned int t = atomic_load_explicit(&ring->tail, memory_order_relaxed);
    unsigned int h = atomic_load_explicit(&ring->head, memory_order_acquire);

    if ((t - h) >= RING_BUFFER_SIZE) {
        return -1; // Queue full
    }

    unsigned int idx = t & (RING_BUFFER_SIZE - 1);
    ring->events[idx].io_u = io_u;
    ring->events[idx].error = error;
    ring->events[idx].bytes_done = bytes_done;

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

/*
 * Completion trampoline. Called from the Python adapter's done_callback, which
 * runs on the asyncio loop thread with the GIL held (ctypes CFUNCTYPE acquires
 * GIL on entry). `ptd_ptr` is the per-job py_thread_data, plumbed through the
 * queue path so each fio job's completions land in its own ring; there is no
 * shared global state, so the engine is safe under numjobs>1 in either process
 * or thread mode.
 */
void c_complete_trampoline(void *ptd_ptr, void *io_u_ptr, int err, unsigned long long bytes_done) {
    struct py_thread_data *ptd = (struct py_thread_data *)ptd_ptr;
    struct io_u *io_u = (struct io_u *)io_u_ptr;

    if (!ptd) {
        fprintf(stderr, "c_complete_trampoline called with NULL ptd!\n");
        return;
    }

    /*
     * Release the GIL across the spin and the eventfd write so the consumer
     * (py_storage_getevents, which runs GIL-free in a fio worker) is never
     * starved by a producer holding the GIL. Today getevents does not need
     * the GIL, but releasing here keeps the engine deadlock-free if that
     * ever changes, and lets other Python threads run while we wait.
     */
    Py_BEGIN_ALLOW_THREADS
    while (ring_push(&ptd->ring, io_u, err, bytes_done) < 0) {
        sched_yield();
    }
    eventfd_t val = 1;
    if (eventfd_write(ptd->event_fd, val) < 0) {
        perror("eventfd_write failed in trampoline");
    }
    Py_END_ALLOW_THREADS
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
/*
 * Initialize the GCSFS lockless async engine. Enforces process-only mode (thread=0).
 */
static int py_storage_init(struct thread_data *td) {
    return 0;
}

static void py_storage_cleanup(struct thread_data *td) {
    struct py_thread_data *ptd = td->io_ops_data;
    if (ptd) {
        if (ptd->event_fd >= 0) {
            close(ptd->event_fd);
        }
        free(ptd);
        td->io_ops_data = NULL;
    }
}

static int run_subprocess_gcs_size(const char *filename, uint64_t *out_size) {
    char command[1024];
    snprintf(command, sizeof(command),
             "python3 -c \"import gcsfs; "
             "fs = gcsfs.GCSFileSystem(); "
             "print(fs.info('%s').get('size', -1))\" 2>/dev/null",
             filename);

    FILE *fp = popen(command, "r");
    if (!fp) {
        return -1;
    }

    char response[128];
    if (fgets(response, sizeof(response), fp) == NULL) {
        pclose(fp);
        return -1;
    }

    int status = pclose(fp);
    if (status != 0) {
        return -1;
    }

    long long size_val = -1;
    if (sscanf(response, "%lld", &size_val) != 1 || size_val < 0) {
        return -1;
    }

    *out_size = (uint64_t)size_val;
    return 0;
}

static int py_init_interpreter_internal(void) {
    if (Py_IsInitialized()) return 0;

    void *libpython = dlopen(PY_SONAME, RTLD_GLOBAL | RTLD_NOW);
    if (!libpython) {
        fprintf(stderr, "Warning: Failed to dlopen %s globally: %s\n", PY_SONAME, dlerror());
    }

    Py_Initialize();

    PyObject *sysPath = PySys_GetObject("path");
    PyObject *cwd = PyUnicode_FromString(".");
    PyList_Append(sysPath, cwd);
    Py_DECREF(cwd);

    PyObject *pName = PyUnicode_FromString("gcsfs_adapter");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "Failed to load module 'gcsfs_adapter'\n");
        return 1;
    }

    pFuncInit = PyObject_GetAttrString(pModule, "py_init");
    pFuncOpen = PyObject_GetAttrString(pModule, "py_open");
    pFuncClose = PyObject_GetAttrString(pModule, "py_close");
    pFuncQueue = PyObject_GetAttrString(pModule, "py_queue");
    pFuncGetSize = PyObject_GetAttrString(pModule, "py_get_file_size");

    if (!pFuncInit || !pFuncOpen || !pFuncClose || !pFuncQueue || !pFuncGetSize) {
        if (PyErr_Occurred()) PyErr_Print();
        fprintf(stderr, "Failed to find required Python functions.\n");
        return 1;
    }

    main_tstate = PyEval_SaveThread();
    return 0;
}

static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
static int py_is_runtime_initialized = 0;

static int py_thread_data_init(struct thread_data *td) {
    if (td->io_ops_data) {
        return 0;
    }

    struct py_thread_data *ptd = calloc(1, sizeof(struct py_thread_data));
    if (!ptd) {
        return 1;
    }

    ptd->event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (ptd->event_fd < 0) {
        perror("Failed to create eventfd");
        free(ptd);
        return 1;
    }

    td->io_ops_data = ptd;
    return 0;
}

static int py_init_runtime_deferred(struct thread_data *td) {
    pthread_mutex_lock(&init_mutex);

    if (py_is_runtime_initialized) {
        pthread_mutex_unlock(&init_mutex);
        return 0;
    }

    if (py_init_interpreter_internal() != 0) {
        pthread_mutex_unlock(&init_mutex);
        return 1;
    }

    // Initialize the Python-side logic (Global Loop & Client)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *arg_iodepth = PyLong_FromLong(td->o.iodepth);
    PyObject *arg_trampoline = PyLong_FromVoidPtr((void *)c_complete_trampoline);
    PyObject *args = PyTuple_Pack(2, arg_iodepth, arg_trampoline);
    Py_XDECREF(arg_iodepth);
    Py_XDECREF(arg_trampoline);
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

    if (success != 0) {
        pthread_mutex_unlock(&init_mutex);
        return 1;
    }

    py_is_runtime_initialized = 1;
    pthread_mutex_unlock(&init_mutex);
    return 0;
}

/*
 * Determine the real file size (using subprocess to avoid contaminating master process).
 */
static int py_storage_get_file_size(struct thread_data *td, struct fio_file *f) {
    uint64_t gcs_size = 0;
    int has_gcs = 0;

    if (run_subprocess_gcs_size(f->file_name, &gcs_size) == 0) {
        has_gcs = 1;
    }

    uint64_t final_size = has_gcs ? gcs_size : 0;
    int found = has_gcs;

    // 2. Override boundaries based on filesize parameter
    if (td->o.file_size_low > final_size) {
        final_size = td->o.file_size_low;
        found = 1;
    }

    // 3. Override boundaries based on job offset/size specs
    uint64_t expected_job_size = td->o.start_offset + td->o.size;
    if (expected_job_size > final_size) {
        final_size = expected_job_size;
        found = 1;
    }

    if (found) {
        if (f->real_file_size == -1ULL || f->real_file_size == 0 || final_size > f->real_file_size) {
            f->real_file_size = final_size;
        }
        fio_file_set_size_known(f);
    }
    return 0;
}

static int py_storage_open(struct thread_data *td, struct fio_file *f) {
    if (py_init_runtime_deferred(td))
        return 1;
    if (py_thread_data_init(td))
        return 1;

    PyGILState_STATE gstate = PyGILState_Ensure();

    struct py_options *o = td->eo;
    int is_write = (td->o.td_ddir == TD_DDIR_WRITE);

    PyObject *arg_filename = PyUnicode_FromString(f->file_name);
    PyObject *arg_is_write = PyBool_FromLong(is_write);
    PyObject *arg_flush = PyBool_FromLong(o->flush_every_write);
    PyObject *arg_size = PyLong_FromLongLong((long long)f->real_file_size);
    PyObject *args = PyTuple_Pack(4, arg_filename, arg_is_write, arg_flush, arg_size);
    Py_XDECREF(arg_filename);
    Py_XDECREF(arg_is_write);
    Py_XDECREF(arg_flush);
    Py_XDECREF(arg_size);

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
    PyObject *arg_handle = PyLong_FromLong(handle);
    PyObject *args = PyTuple_Pack(1, arg_handle);
    Py_XDECREF(arg_handle);
    PyObject *result = PyObject_CallObject(pFuncClose, args);
    Py_DECREF(args);

    if (result) Py_DECREF(result);
    else PyErr_Print();

    PyGILState_Release(gstate);
    return 0;
}

static int py_gcsfs_io_u_init(struct thread_data *td, struct io_u *io_u) {
    struct gcsfs_io_u_data *iud = calloc(1, sizeof(*iud));
    if (!iud) return 1;
    io_u->engine_data = iud;
    return 0;
}

static void py_gcsfs_io_u_free(struct thread_data *td, struct io_u *io_u) {
    struct gcsfs_io_u_data *iud = io_u->engine_data;
    if (iud) {
        if (iud->memview) {
            PyGILState_STATE gstate = PyGILState_Ensure();
            Py_DECREF((PyObject *)iud->memview);
            PyGILState_Release(gstate);
        }
        free(iud);
        io_u->engine_data = NULL;
    }
}

static enum fio_q_status py_storage_queue(struct thread_data *td, struct io_u *io_u) {
    struct py_thread_data *ptd = td->io_ops_data;
    if (!ptd) {
        io_u->error = EIO;
        return FIO_Q_COMPLETED;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    long handle = (long)(uintptr_t)io_u->file->engine_data;
    int is_write = (io_u->ddir == DDIR_WRITE);
    struct gcsfs_io_u_data *iud = io_u->engine_data;

    // Zero-Copy Magic: Create/reuse a MemoryView directly on the FIO C buffer.
    if (!iud->memview || iud->cached_buf != io_u->xfer_buf || iud->cached_len != io_u->xfer_buflen) {
        if (iud->memview) {
            Py_DECREF((PyObject *)iud->memview);
        }
        iud->memview = PyMemoryView_FromMemory(
            (char *)io_u->xfer_buf,
            io_u->xfer_buflen,
            is_write ? PyBUF_READ : PyBUF_WRITE
        );
        if (!iud->memview) {
            PyErr_Print();
            PyGILState_Release(gstate);
            io_u->error = EIO;
            return FIO_Q_COMPLETED;
        }
        iud->cached_buf = io_u->xfer_buf;
        iud->cached_len = io_u->xfer_buflen;
    }

    // Pass the io_u pointer as the per-IO 'tag' and the per-job ptd pointer so
    // the completion trampoline knows which ring to push into. Both are opaque
    // PyLong-wrapped pointers from Python's perspective.
    PyObject *arg_handle = PyLong_FromLong(handle);
    PyObject *arg_ptd = PyLong_FromVoidPtr(ptd);
    PyObject *arg_tag = PyLong_FromVoidPtr(io_u);
    PyObject *arg_offset = PyLong_FromLongLong(io_u->offset);
    PyObject *arg_is_write = PyBool_FromLong(is_write);
    PyObject *args = PyTuple_Pack(6,
        arg_handle,
        arg_ptd,
        arg_tag,
        arg_offset,
        (PyObject *)iud->memview,
        arg_is_write
    );
    Py_XDECREF(arg_handle);
    Py_XDECREF(arg_ptd);
    Py_XDECREF(arg_tag);
    Py_XDECREF(arg_offset);
    Py_XDECREF(arg_is_write);

    PyObject *result = PyObject_CallObject(pFuncQueue, args);
    Py_DECREF(args);

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
    unsigned long long bytes_done = ptd->events[event].bytes_done;

    if (err != 0) {
        io_u->error = EIO;
    } else {
        if (bytes_done <= io_u->xfer_buflen) {
            io_u->resid = io_u->xfer_buflen - bytes_done;
        } else {
            io_u->resid = 0;
        }
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
    .get_file_size = py_storage_get_file_size,
    .queue = py_storage_queue,
    .getevents = py_storage_getevents,
    .event = py_storage_event,
    .io_u_init = py_gcsfs_io_u_init,
    .io_u_free = py_gcsfs_io_u_free,
    .flags = FIO_DISKLESSIO | FIO_NOEXTEND | FIO_NODISKUTIL,
    .options = options,
    .option_struct_size = sizeof(struct py_options),
};
