#!/bin/bash
# Native (no-Docker) build of MaxVec/Maximus on Euler, mirroring maxvec/Dockerfile.
# Everything is built on the node-local disk under a fixed path (/tmp/$USER/maxvec) because
# home (CephFS) is too slow for many small writes. The result is packed into ONE archive in
# ~/maxvec-snapshots/; ~/maxvec-env.sh unpacks it to the same fixed path on any node.
# Each step writes a stamp file, so rerunning resumes where it stopped; on a fresh node it
# first unpacks the saved snapshot, so a rerun = rebuild MaxVec from $REPO + repack.
#
#   sbatch ~/vecopt/euler/build_maxvec_euler.sh   (log lands in the submit dir)
#   use afterwards:  source ~/maxvec-env.sh
#
#SBATCH --job-name=maxvec-build
#SBATCH --partition=research
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

ROOT=/tmp/$USER/maxvec                   # fixed path, identical on every node
ENV_PREFIX=$ROOT/env
DEPS=$ROOT/deps                          # faiss source, stamps
MINIFORGE=$ROOT/miniforge3
BUILD_DIR_ROOT=$ROOT/build               # Maximus build dirs (source stays in home)
SNAP_DIR=$HOME/maxvec-snapshots
REPO=${REPO:-$HOME/vecopt/maxvec}
JOBS=${SLURM_CPUS_PER_TASK:-16}
BUILD_TYPE=${BUILD_TYPE:-Release}
# Compute nodes mix AMD Milan/Rome (no AVX-512) with Genoa/Sapphire Rapids: avx2 runs everywhere.
OPT_LEVEL=${OPT_LEVEL:-avx2}
# A100=80, RTX A4500=86, RTX 4000 Ada=89, H100=90
CUDA_ARCHS=${CUDA_ARCHS:-"80;86;89;90"}
TMPBUILD=$ROOT/tmp
mkdir -p "$DEPS/stamps" "$TMPBUILD" "$SNAP_DIR"
export CONDA_PKGS_DIRS=$ROOT/pkgs TMPDIR=$TMPBUILD

# Fresh node: start from the saved snapshot, so finished steps (stamped) are skipped.
if [ ! -d "$ENV_PREFIX" ] && [ -f "$SNAP_DIR/maxvec-env.tar.zst" ]; then
    echo "== unpack snapshot [$(date +%T)]"
    tar -C /tmp/$USER -I zstd -xf "$SNAP_DIR/maxvec-env.tar.zst"
fi

step() { [ -f "$DEPS/stamps/$1" ] && { echo "== skip $1 (done)"; return 1; }; echo "== $1 [$(date +%T)]"; }
done_() { touch "$DEPS/stamps/$1"; }

# Own Miniforge: the system one (24.3) has an unwritable shared pkg cache
if [ ! -x "$MINIFORGE/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$TMPBUILD/mf.sh"
    bash "$TMPBUILD/mf.sh" -b -p "$MINIFORGE"
fi
eval "$("$MINIFORGE/bin/conda" shell.bash hook)"
activate() { set +u; conda activate "$1"; set -u; }

# 1. Conda env (same package set as the Dockerfile + CUDA toolchain, which the Docker base image provided)
if step conda-env; then
    if [ -f "$ENV_PREFIX/conda-meta/history" ] && ls "$ENV_PREFIX"/conda-meta/libcuvs-25.08* >/dev/null 2>&1; then
        echo "env already created, reusing"
    else
    rm -rf "$ENV_PREFIX"
    mamba create -p "$ENV_PREFIX" -y --override-channels \
        -c rapidsai -c conda-forge \
        python=3.10 cuda-version=12.8 \
        cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvtx-dev cuda-profiler-api \
        "libcuvs=25.08" "libraft=25.08" "cudf=25.08" \
        "gcc_linux-64=13" "gxx_linux-64=13" sysroot_linux-64 \
        gdb cmake ninja pkg-config swig sqlite \
        "openblas=*=openmp*" glog gflags gtest gmock boost-cpp rapidjson \
        brotli snappy zlib lz4-c zstd libprotobuf \
        pandas pyarrow duckdb
    fi
    activate "$ENV_PREFIX"
    pip install --no-cache-dir psycopg pgvector tabulate openpyxl matplotlib seaborn plotly
    done_ conda-env
fi
activate "$ENV_PREFIX"
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CUDAToolkit_ROOT=$CONDA_PREFIX
echo "CC=$CC CXX=$CXX nvcc=$(nvcc --version | tail -1)"

# 2. rapids-cmake
if step rapids-cmake; then
    rm -rf "$TMPBUILD/rc"
    git clone --depth 1 --branch branch-25.08 https://github.com/rapidsai/rapids-cmake.git "$TMPBUILD/rc"
    cmake -S "$TMPBUILD/rc" -B "$TMPBUILD/rc/build" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    cmake --install "$TMPBUILD/rc/build"
    done_ rapids-cmake
fi

# 3. Arrow 21 (flags copied from the Dockerfile)
if step arrow; then
    rm -rf "$TMPBUILD/arrow"
    git clone --depth 1 --recursive --branch apache-arrow-21.0.0 https://github.com/apache/arrow "$TMPBUILD/arrow"
    cmake -S "$TMPBUILD/arrow/cpp" -B "$TMPBUILD/arrow/cpp/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -Dxsimd_SOURCE=BUNDLED -DARROW_WITH_UTF8PROC=OFF -DARROW_WITH_RE2=ON \
        -DARROW_FILESYSTEM=ON -DARROW_TESTING=OFF -DARROW_WITH_LZ4=OFF -DARROW_WITH_ZSTD=OFF \
        -DARROW_BUILD_STATIC=ON -DARROW_BUILD_SHARED=ON -DARROW_BUILD_TESTS=OFF \
        -DARROW_BUILD_BENCHMARKS=OFF -DARROW_IPC=ON -DARROW_FLIGHT=OFF -DARROW_COMPUTE=ON \
        -DARROW_CUDA=OFF -DARROW_JEMALLOC=OFF -DARROW_USE_GLOG=OFF -DARROW_DATASET=ON \
        -DARROW_BUILD_UTILITIES=OFF -DARROW_HDFS=OFF -DARROW_TENSORFLOW=OFF \
        -DARROW_CSV=ON -DARROW_JSON=ON -DARROW_WITH_BROTLI=ON -DARROW_WITH_SNAPPY=ON \
        -DARROW_WITH_ZLIB=ON -DARROW_PARQUET=ON -DARROW_SUBSTRAIT=OFF -DARROW_ACERO=ON \
        -DARROW_WITH_BACKTRACE=ON -DARROW_CXXFLAGS="-w" -DARROW_S3=OFF -DARROW_ORC=OFF \
        -DARROW_POSITION_INDEPENDENT_CODE=ON -DARROW_DEPENDENCY_USE_SHARED=ON \
        -DARROW_BOOST_USE_SHARED=ON -DARROW_BROTLI_USE_SHARED=ON -DARROW_GFLAGS_USE_SHARED=ON \
        -DARROW_GRPC_USE_SHARED=ON -DARROW_PROTOBUF_USE_SHARED=ON -DARROW_ZSTD_USE_SHARED=ON \
        -DProtobuf_SOURCE=SYSTEM
    ninja -C "$TMPBUILD/arrow/cpp/build" -j $JOBS install
    rm -rf "$TMPBUILD/arrow"
    done_ arrow
fi

# 4. Taskflow 3.6
if step taskflow; then
    rm -rf "$TMPBUILD/taskflow"
    git clone --depth 1 --branch v3.6.0 https://github.com/taskflow/taskflow.git "$TMPBUILD/taskflow"
    cmake -S "$TMPBUILD/taskflow" -B "$TMPBUILD/taskflow/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DTF_BUILD_TESTS=OFF -DTF_BUILD_EXAMPLES=OFF
    cmake --install "$TMPBUILD/taskflow/build"
    done_ taskflow
fi

# 5. Caliper (NVTX on, CUDA off)
if step caliper; then
    rm -rf "$TMPBUILD/caliper"
    git clone --depth 1 https://github.com/LLNL/Caliper.git "$TMPBUILD/caliper"
    cmake -S "$TMPBUILD/caliper" -B "$TMPBUILD/caliper/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -DWITH_CUDA=OFF -DCUDAToolkit_ROOT=$CONDA_PREFIX -DWITH_NVTX=ON \
        -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF
    ninja -C "$TMPBUILD/caliper/build" -j $JOBS install
    grep -q '^#define CALIPER_HAVE_NVTX' $CONDA_PREFIX/include/caliper/caliper-config.h
    done_ caliper
fi

# 6. raft ATS patch (header-only, must precede Faiss)
if step raft-patch; then
    (cd $CONDA_PREFIX/include && patch -p0 --forward < "$REPO/third_party/patches/raft-ats-runtime.patch")
    done_ raft-patch
fi

# 7. Faiss 1.13 + cuVS, with CAGRA-graph and IVF-host-view patches. Source kept in $DEPS/faiss for rebuilds.
if step faiss; then
    rm -rf "$DEPS/faiss"
    git clone --depth 1 --branch v1.13.0 https://github.com/facebookresearch/faiss.git "$DEPS/faiss"
    (cd "$DEPS/faiss" && patch -p0 --forward < "$REPO/third_party/patches/faiss-cagra-graph.patch" \
                      && patch -p0 --forward < "$REPO/third_party/patches/faiss-ivf-host-view.patch")
    cmake -S "$DEPS/faiss" -B "$DEPS/faiss/build" -G Ninja \
        -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_CUVS=ON -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_OPT_LEVEL=$OPT_LEVEL -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DBLA_VENDOR=OpenBLAS -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    ninja -C "$DEPS/faiss/build" -j $JOBS faiss swigfaiss
    ninja -C "$DEPS/faiss/build" install
    (cd "$DEPS/faiss/build/faiss/python" && pip install --no-build-isolation .)
    done_ faiss
fi

# 8. Maximus (same flags as `make build-maximus`); rebuilt every run
echo "== maximus [$(date +%T)]"
# Build dir configured for a different source tree (e.g. repo moved): start it fresh.
CACHE="$BUILD_DIR_ROOT/$BUILD_TYPE/CMakeCache.txt"
if [ -f "$CACHE" ] && ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$(realpath "$REPO")" "$CACHE"; then
    echo "build dir points at another source tree, recreating"
    rm -rf "$BUILD_DIR_ROOT/$BUILD_TYPE"
fi
# Static libfaiss contains CUB/Thrust kernel instances that also exist in libcuvs.so. If they are
# exported, libcuvs's calls bind to our copies -> "cudaErrorInvalidDeviceFunction" in cuVS k-means
# (CAGRA/IVF builds). Keep Faiss's symbols private. Keep conda's $LDFLAGS (rpath-link etc.).
FAISS_HIDE="${LDFLAGS:-} -Wl,--exclude-libs,libfaiss.a:libfaiss_avx2.a"
cmake -S "$REPO" -B "$BUILD_DIR_ROOT/$BUILD_TYPE" -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMAXIMUS_WITH_TESTS=ON -DMAXIMUS_WITH_GPU=ON -DMAXIMUS_WITH_FAISS=GPUCUVS \
    -DMAXIMUS_WITH_VS=ON -DMAXIMUS_WITH_PROFILING=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_SHARED_LINKER_FLAGS="$FAISS_HIDE" -DCMAKE_EXE_LINKER_FLAGS="$FAISS_HIDE"
ninja -C "$BUILD_DIR_ROOT/$BUILD_TYPE" -j $JOBS

# Pack into one archive on home (one big sequential write instead of ~100k small ones).
# Faiss object files, the pkg cache and temp files are left out.
echo "== pack [$(date +%T)] $(du -sh --exclude=pkgs --exclude=tmp --exclude=build "$ROOT" | cut -f1) before compression"
rm -rf "$ROOT/pkgs" "$TMPBUILD"/*
tar -C /tmp/$USER --exclude=maxvec/deps/faiss/build --exclude=maxvec/tmp \
    -I "zstd -T$JOBS -6" -cf "$SNAP_DIR/maxvec-env.tar.zst.part" maxvec
zstd -t -q "$SNAP_DIR/maxvec-env.tar.zst.part"
mv -f "$SNAP_DIR/maxvec-env.tar.zst.part" "$SNAP_DIR/maxvec-env.tar.zst"
ls -la "$SNAP_DIR"

# Env file for interactive use / job scripts
cat > $HOME/maxvec-env.sh <<'EOF'
# source ~/maxvec-env.sh  -- on any node: unpack MaxVec env to node-local disk if missing, then activate
MAXVEC_ROOT=/tmp/$USER/maxvec
MAXVEC_SNAP=$HOME/maxvec-snapshots/maxvec-env.tar.zst
mkdir -p /tmp/$USER
(
  flock 9
  if [ ! -x "$MAXVEC_ROOT/env/bin/python" ] || [ "$MAXVEC_SNAP" -nt "$MAXVEC_ROOT/.unpacked" ]; then
    echo "[maxvec-env] unpacking $MAXVEC_SNAP -> $MAXVEC_ROOT (once per node)..."
    rm -rf "$MAXVEC_ROOT" && tar -C /tmp/$USER -I zstd -xf "$MAXVEC_SNAP" && touch "$MAXVEC_ROOT/.unpacked"
  fi
) 9>/tmp/$USER/.maxvec.lock
# conda activation scripts read unset variables: allow that even if the caller uses `set -u`
case $- in *u*) _maxvec_u=1; set +u;; *) _maxvec_u=;; esac
eval "$($MAXVEC_ROOT/miniforge3/bin/conda shell.bash hook)"
conda activate $MAXVEC_ROOT/env
[ -n "$_maxvec_u" ] && set -u; unset _maxvec_u
export MAXVEC_BUILD=$MAXVEC_ROOT/build/Release
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CUDAToolkit_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
EOF
touch "$ROOT/.unpacked"
echo "== ALL DONE [$(date +%T)] -> source ~/maxvec-env.sh ; binaries in $BUILD_DIR_ROOT/$BUILD_TYPE"
