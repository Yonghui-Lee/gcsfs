#!/usr/bin/env bash
# ==============================================================================
# GCSFS FIO Benchmark Wrapper Script
# ==============================================================================
# This script automatically detects the active Python runtime's shared library
# (libpython) and configures the environment (PYTHONPATH, LD_PRELOAD) so that
# the CPython-embedded GCSFS FIO engines can run plug-and-play on any system.
# ==============================================================================

set -eo pipefail

# 1. Determine script directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 2. Check if FIO is installed
if ! command -v fio &> /dev/null; then
    echo -e "\033[31;1mError: 'fio' command not found on this system.\033[0m"
    echo -e "Please install FIO and development libraries first. On Debian/Ubuntu:"
    echo -e "  sudo apt-get update && sudo apt-get install -y fio"
    exit 1
fi

# 3. Check if the FIO engines are compiled
if [ ! -f "$SCRIPT_DIR/libgcsfs_async_fio_engine.so" ] || [ ! -f "$SCRIPT_DIR/libgcsfs_sync_fio_engine.so" ]; then
    echo -e "\033[33;1mWarning: Embedded GCSFS engines are not compiled or missing.\033[0m"
    echo -e "Attempting to build them now using 'make'..."
    make -C "$SCRIPT_DIR"
    echo -e "\033[32;1mBuild complete!\033[0m"
fi

# 4. Resolve the active Python dynamic shared library path
echo -e "\033[34m[run.sh] Auto-detecting active Python shared library...\033[0m"
DETECTED_LIBPYTHON=$(python3 -c "
import sysconfig
from pathlib import Path

def get_libpython():
    libdir = sysconfig.get_config_var('LIBDIR')
    ldlibrary = sysconfig.get_config_var('LDLIBRARY')
    instsoname = sysconfig.get_config_var('INSTSONAME')

    candidates = []
    if libdir:
        if ldlibrary: candidates.append(Path(libdir) / ldlibrary)
        if instsoname: candidates.append(Path(libdir) / instsoname)

    prefix = sysconfig.get_config_var('prefix')
    if prefix:
        libpath = Path(prefix) / 'lib'
        if libpath.exists():
            for ext in ('.so', '.so.1.0', '.dylib'):
                for p in libpath.glob(f'libpython*{ext}'):
                    candidates.append(p)

    valid_candidates = []
    for c in candidates:
        try:
            c = c.resolve()
            if c.is_file() and not c.name.endswith('.a') and c not in valid_candidates:
                valid_candidates.append(c)
        except Exception:
            pass

    if valid_candidates:
        for c in valid_candidates:
            if instsoname and c.name == instsoname: return str(c)
        for c in valid_candidates:
            if ldlibrary and c.name == ldlibrary: return str(c)
        return str(valid_candidates[0])
    return None

print(get_libpython() or 'NOT_FOUND')
")

if [ "$DETECTED_LIBPYTHON" = "NOT_FOUND" ] || [ -z "$DETECTED_LIBPYTHON" ]; then
    echo -e "\033[33;1mWarning: Could not auto-detect active libpython shared library path.\033[0m"
    echo -e "Your Python environment might be statically linked or compiled without --enable-shared."
    echo -e "We will attempt to run FIO without preloading libpython..."
else
    echo -e "\033[32m[run.sh] Found active libpython: $DETECTED_LIBPYTHON\033[0m"
    # Set LD_PRELOAD unless the user has already overridden it
    if [ -z "${LD_PRELOAD}" ]; then
        export LD_PRELOAD="$DETECTED_LIBPYTHON"
    else
        echo -e "\033[34m[run.sh] Respecting user-specified LD_PRELOAD=$LD_PRELOAD\033[0m"
    fi
fi

# 5. Validate BUCKET_NAME environment variable when running job specs that require it
NEEDS_BUCKET=false
for arg in "$@"; do
    if [[ "$arg" == *.fio ]] && [ -f "$arg" ]; then
        if grep -q "BUCKET_NAME" "$arg"; then
            NEEDS_BUCKET=true
            break
        fi
    fi
done

if [ "$NEEDS_BUCKET" = true ] && [ -z "${BUCKET_NAME}" ]; then
    echo -e "\033[31;1mError: The 'BUCKET_NAME' environment variable is not set but is required by the target job.\033[0m"
    echo -e "To target a specific Google Cloud Zonal (or regional) bucket, please export it first:"
    echo -e "  export BUCKET_NAME=\"my-zonal-bucket-name\""
    echo -e "Or run inline:"
    echo -e "  BUCKET_NAME=\"my-zonal-bucket-name\" ./run.sh jobs/smoke_test_read.fio"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Print details of the run
echo -e "\033[34m[run.sh] Environment configured:\033[0m"
echo -e "  - PYTHONPATH: $PYTHONPATH"
echo -e "  - BUCKET_NAME: ${BUCKET_NAME}"
if [ -n "${STORAGE_EMULATOR_HOST}" ]; then
    echo -e "  - STORAGE_EMULATOR_HOST: ${STORAGE_EMULATOR_HOST}"
fi
echo -e "  - Running command: fio $*\033[0m\n"

# 6. Execute FIO with all original command line arguments
exec fio "$@"
