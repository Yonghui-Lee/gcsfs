# prototype/Makefile

FIO_SRC ?= /usr/include
PY_CFLAGS = $(shell python3-config --cflags --embed)
PY_LDFLAGS = $(shell python3-config --ldflags --embed)

libgcsfs_fio_engine.so: gcsfs_engine.c
	gcc -O3 -g -shared -fPIC -Wall -Wextra \
		-o libgcsfs_fio_engine.so gcsfs_engine.c \
		-I$(FIO_SRC) \
		$(PY_CFLAGS) $(PY_LDFLAGS) \
		-pthread

clean:
	rm -f libgcsfs_fio_engine.so
