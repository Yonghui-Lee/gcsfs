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
static PyObject *pFuncGetSize = NULL;

static PyThreadState *main_tstate = NULL;
static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * FIO custom options specifically for the synchronous engine
 */
struct py_sync_options {
    void *pad;
    unsigned int block_size;
    unsigned int use_prefetch;
    unsigned int concurrency;
    char *cache_type;
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
        .name = "cache_type",
        .lname = "cache_type",
        .type = FIO_OPT_STR_STORE,
        .off1 = offsetof(struct py_sync_options, cache_type),
        .help = "GCSFS cache strategy (e.g. none, readahead, readahead_chunked)",
        .category = FIO_OPT_C_ENGINE,
        .group = FIO_OPT_G_INVALID,
    },
    {
        .name = NULL,
    },
};

/*
 * Initialize the GCSFS sync engine. Enforces process-only model (thread=0).
 */
static int py_sync_storage_init(struct thread_data *td) {
    return 0;
}

static void py_sync_storage_cleanup(struct thread_data *td) {
    // No-op. Option strings are freed by FIO.
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

static int py_init_interpreter_internal(struct thread_data *td) {
    if (Py_IsInitialized()) return 0;

    struct py_sync_options *o = td->eo;
    char buf[32];
    snprintf(buf, sizeof(buf), "%u", o ? o->concurrency : 4);
    setenv("DEFAULT_GCSFS_CONCURRENCY", buf, 1);

    void *libpython = dlopen(PY_SONAME, RTLD_GLOBAL | RTLD_NOW);
    if (!libpython) {
        fprintf(stderr, "Warning: Failed to dlopen %s globally: %s\n", PY_SONAME, dlerror());
    }

    Py_Initialize();

    PyObject *sysPath = PySys_GetObject("path");
    PyObject *cwd = PyUnicode_FromString(".");
    PyList_Append(sysPath, cwd);
    Py_DECREF(cwd);

    PyObject *pName = PyUnicode_FromString("gcsfs_sync_adapter");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "Failed to load Python module 'gcsfs_sync_adapter'\n");
        return 1;
    }

    pFuncInit = PyObject_GetAttrString(pModule, "py_sync_init");
    pFuncOpen = PyObject_GetAttrString(pModule, "py_sync_open");
    pFuncClose = PyObject_GetAttrString(pModule, "py_sync_close");
    pFuncRead = PyObject_GetAttrString(pModule, "py_sync_read");
    pFuncWrite = PyObject_GetAttrString(pModule, "py_sync_write");
    pFuncGetSize = PyObject_GetAttrString(pModule, "py_sync_get_file_size");

    if (!pFuncInit || !pFuncOpen || !pFuncClose || !pFuncRead || !pFuncWrite || !pFuncGetSize) {
        if (PyErr_Occurred()) PyErr_Print();
        fprintf(stderr, "Failed to resolve required python callbacks in gcsfs_sync_adapter.\n");
        return 1;
    }

    main_tstate = PyEval_SaveThread();
    return 0;
}

static int py_is_runtime_initialized = 0;

static int py_sync_init_runtime_deferred(struct thread_data *td) {
    pthread_mutex_lock(&init_mutex);

    if (py_is_runtime_initialized) {
        pthread_mutex_unlock(&init_mutex);
        return 0;
    }

    if (py_init_interpreter_internal(td) != 0) {
        pthread_mutex_unlock(&init_mutex);
        return 1;
    }

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
static int py_sync_storage_get_file_size(struct thread_data *td, struct fio_file *f) {
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

/*
 * Open the target file. Invoked during FIO start or preparation.
 */
static int py_sync_storage_open(struct thread_data *td, struct fio_file *f) {
    if (py_sync_init_runtime_deferred(td))
        return 1;

    struct py_sync_options *o = td->eo;
    int is_write = (td->o.td_ddir == TD_DDIR_WRITE);

    PyGILState_STATE gstate = PyGILState_Ensure();

    // Invoke open callback passing open context parameters. cache_type_obj is
    // always a strong reference here: either a fresh PyUnicode or Py_None
    // with a matching INCREF, so the unconditional Py_DECREF below balances.
    PyObject *cache_type_obj = NULL;
    if (o->cache_type) {
        cache_type_obj = PyUnicode_FromString(o->cache_type);
    }
    if (!cache_type_obj) {
        cache_type_obj = Py_None;
        Py_INCREF(Py_None);
    }

    PyObject *arg_filename = PyUnicode_FromString(f->file_name);
    PyObject *arg_is_write = PyBool_FromLong(is_write);
    PyObject *arg_block_size = PyLong_FromLong(o->block_size);
    PyObject *arg_use_prefetch = PyBool_FromLong(o->use_prefetch);
    PyObject *arg_concurrency = PyLong_FromLong(o->concurrency);
    PyObject *args = PyTuple_Pack(6,
        arg_filename,
        arg_is_write,
        arg_block_size,
        arg_use_prefetch,
        arg_concurrency,
        cache_type_obj
    );
    Py_XDECREF(arg_filename);
    Py_XDECREF(arg_is_write);
    Py_XDECREF(arg_block_size);
    Py_XDECREF(arg_use_prefetch);
    Py_XDECREF(arg_concurrency);
    Py_DECREF(cache_type_obj);

    PyObject *result = PyObject_CallObject(pFuncOpen, args);
    Py_DECREF(args);

    if (result == NULL) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return 1;
    }

    if (result == Py_None) {
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return 1;
    }

    // Keep handle registration in FIO file context
    f->engine_data = (void *)result;

    PyGILState_Release(gstate);
    return 0;
}

/*
 * Close the target file.
 */
static int py_sync_storage_close(struct thread_data *td, struct fio_file *f) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *handle_obj = (PyObject *)f->engine_data;
    if (handle_obj) {
        PyObject *result = PyObject_CallFunctionObjArgs(pFuncClose, handle_obj, NULL);
        if (result) Py_DECREF(result);
        else PyErr_Print();
        Py_DECREF(handle_obj); // Release reference kept since open
        f->engine_data = NULL;
    }

    PyGILState_Release(gstate);
    return 0;
}

/*
 * Synchronous queue method. Excutes block I/O completely synchronously.
 */
static enum fio_q_status py_sync_storage_queue(struct thread_data *td, struct io_u *io_u) {
    PyObject *handle_obj = (PyObject *)io_u->file->engine_data;
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

    PyObject *arg_offset = PyLong_FromLongLong(io_u->offset);

    PyObject *result;
    if (is_write) {
        result = PyObject_CallFunctionObjArgs(pFuncWrite, handle_obj, arg_offset, memview, NULL);
    } else {
        result = PyObject_CallFunctionObjArgs(pFuncRead, handle_obj, arg_offset, memview, NULL);
    }

    Py_XDECREF(arg_offset);
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
    .get_file_size = py_sync_storage_get_file_size,
    .queue = py_sync_storage_queue,
    .flags = FIO_DISKLESSIO | FIO_NOEXTEND | FIO_NODISKUTIL | FIO_SYNCIO,
    .options = options,
    .option_struct_size = sizeof(struct py_sync_options),
};
