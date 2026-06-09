# fio/Makefile
# ==============================================================================
# GCSFS FIO Embedded Engines Build & Test Suite
# ==============================================================================

FIO_VERSION = 3.41
FIO_SRC = fio-src
PY_CFLAGS = $(shell python3-config --cflags --embed)
PY_LDFLAGS = $(shell python3-config --ldflags --embed)
PY_SONAME = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('INSTSONAME'))")

.PHONY: all clean clean-all help check-bucket run-smoke-async run-smoke-sync run-smoke-local run-smoke-all

# Default target compiles all engines
all: libgcsfs_fio_engine.so libgcsfs_sync_fio_engine.so

libgcsfs_fio_engine.so: gcsfs_engine.c $(FIO_SRC)/config-host.h
	gcc -O3 -g -shared -fPIC -Wall -Wextra -Wno-unused-parameter \
		-DPY_SONAME=\"$(PY_SONAME)\" \
		-o libgcsfs_fio_engine.so gcsfs_engine.c \
		-I$(FIO_SRC) \
		$(PY_CFLAGS) $(PY_LDFLAGS) \
		-pthread

libgcsfs_sync_fio_engine.so: gcsfs_sync_engine.c $(FIO_SRC)/config-host.h
	gcc -O3 -g -shared -fPIC -Wall -Wextra -Wno-unused-parameter \
		-DPY_SONAME=\"$(PY_SONAME)\" \
		-o libgcsfs_sync_fio_engine.so gcsfs_sync_engine.c \
		-I$(FIO_SRC) \
		$(PY_CFLAGS) $(PY_LDFLAGS) \
		-pthread

$(FIO_SRC):
	mkdir -p $(FIO_SRC)
	@echo "Downloading FIO $(FIO_VERSION) headers..."
	curl -L https://github.com/axboe/fio/archive/refs/tags/fio-$(FIO_VERSION).tar.gz | tar -xz -C $(FIO_SRC) --strip-components=1

$(FIO_SRC)/config-host.h: $(FIO_SRC)
	@echo "Configuring FIO environment..."
	cd $(FIO_SRC) && ./configure

clean:
	rm -f libgcsfs_fio_engine.so libgcsfs_sync_fio_engine.so

clean-all: clean
	rm -rf $(FIO_SRC)

# ------------------------------------------------------------------------------
# Validation Rules
# ------------------------------------------------------------------------------

check-bucket:
	@if [ -z "$$BUCKET_NAME" ]; then \
		echo -e "\033[31;1mError: The 'BUCKET_NAME' environment variable is not set.\033[0m"; \
		echo -e "Please set your Google Cloud Zonal (or regional) bucket name before running benchmarks:"; \
		echo -e "  export BUCKET_NAME=\"my-zonal-bucket\""; \
		exit 1; \
	fi

# ------------------------------------------------------------------------------
# Smoke Testing Targets (Runs against real GCS/Zonal buckets)
# ------------------------------------------------------------------------------

run-smoke-async: all check-bucket
	@echo "=============================================================================="
	@echo " Running Lockless Async Engine Smoke Tests on gs://$(BUCKET_NAME)"
	@echo "=============================================================================="
	./run.sh jobs/smoke_test_write.fio
	./run.sh jobs/smoke_test_read.fio

run-smoke-sync: all check-bucket
	@echo "=============================================================================="
	@echo " Running Pure Synchronous Engine Smoke Tests on gs://$(BUCKET_NAME)"
	@echo "=============================================================================="
	./run.sh jobs/smoke_test_sync_write.fio
	./run.sh jobs/smoke_test_sync_read.fio
	./run.sh jobs/smoke_test_sync_read_readahead.fio

run-smoke-local: all
	@echo "=============================================================================="
	@echo " Running Local Storage Engine Smoke Tests (No cloud bucket required)"
	@echo "=============================================================================="
	./run.sh jobs/smoke_test_sync_write_local.fio
	./run.sh jobs/smoke_test_sync_read_local.fio

run-smoke-all: run-smoke-async run-smoke-sync
	@echo "=============================================================================="
	@echo " All smoke tests completed successfully on gs://$(BUCKET_NAME)!"
	@echo "=============================================================================="

# ------------------------------------------------------------------------------
# Help UI
# ------------------------------------------------------------------------------

help:
	@echo "=============================================================================="
	@echo " GCSFS FIO Embedded Engine Build & Test System"
	@echo "=============================================================================="
	@echo "Build Commands:"
	@echo "  make                 Compile both embedded engines (async and sync)"
	@echo "  make clean           Remove compiled shared engines"
	@echo "  make clean-all       Remove compiled engines and downloaded FIO headers"
	@echo ""
	@echo "Benchmark Commands:"
	@echo "  make run-smoke-all   Run all cloud write/read smoke tests (requires BUCKET_NAME)"
	@echo "  make run-smoke-async Run only the lockless async engine cloud smoke tests"
	@echo "  make run-smoke-sync  Run only the pure synchronous engine cloud smoke tests"
	@echo "  make run-smoke-local Run the local storage synchronous engine smoke tests (no cloud required)"
	@echo "=============================================================================="
