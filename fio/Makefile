# fio/Makefile

FIO_VERSION = 3.41
FIO_SRC = fio-src
PY_CFLAGS = $(shell python3-config --cflags --embed)
PY_LDFLAGS = $(shell python3-config --ldflags --embed)

libgcsfs_fio_engine.so: gcsfs_engine.c $(FIO_SRC)/config-host.h
	gcc -O3 -g -shared -fPIC -Wall -Wextra \
		-o libgcsfs_fio_engine.so gcsfs_engine.c \
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
	rm -f libgcsfs_fio_engine.so

clean-all: clean
	rm -rf $(FIO_SRC)
