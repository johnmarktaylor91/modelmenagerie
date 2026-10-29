# Build a committed environment lock on the cluster (CentOS 7, glibc 2.17), once per lock.
# Usage (from cluster.job_script, in the code checkout): bash envs/axon/build.sh NAME PREFIX LOCK_ID [SKIP...]
#
# `uv sync --frozen` of the same lock the worker uses, with two differences, both written to
# PREFIX/.menagerie-cluster: the SKIP packages (the renderer's binary-only ones; the cluster never
# renders) are left out, and the locked packages whose wheels all need a newer glibc (numpy,
# pillow, contourpy at the torch-main pin) build from their locked sdists with the gcc 10 module,
# a static C++ runtime and C libraries from a conda-forge sysroot (rpath-linked, so attempts need
# no library path). torch 2.6's CPU wheel itself needs only GLIBC_2.17 and GLIBCXX_3.4.19.
set -euo pipefail
name=$1 prefix=$2 lock=$3; shift 3
if [ "$(cat "$prefix/.menagerie-lock" 2>/dev/null)" = "$lock" ]; then
  echo "built: $lock"; exit 0
fi
root=$(dirname "$(dirname "$prefix")")  # ROOT/envs/<prefix> -> ROOT
# C libraries only: a conda-forge C++ library would need a newer libstdc++ than the system's,
# and the process loads one libstdc++ (torch's needs are met by the system one). numpy builds
# without an external BLAS; pillow without TIFF, JPEG 2000 and colour management.
libs="zlib libjpeg-turbo libpng freetype libwebp-base pkg-config"
sysroot=$root/sysroot-$(printf %s "$libs" | sha256sum | cut -c1-8)
if [ ! -f "$sysroot/.complete" ]; then
  CONDA_PKGS_DIRS=$root/cache/conda /share/apps/Miniforge/bin/conda create -y -q -p "$sysroot" \
    --override-channels -c conda-forge $libs
  touch "$sysroot/.complete"
fi
set +u; source /etc/profile.d/z00-lmod.sh && module load gcc/10.4; set -u
export CC=$(command -v gcc) CXX=$(command -v g++)
export PKG_CONFIG_PATH=$sysroot/lib/pkgconfig CFLAGS="-I$sysroot/include" CPPFLAGS="-I$sysroot/include"
# DT_RPATH (not RUNPATH), so the sysroot libraries' own dependencies resolve there too.
export LDFLAGS="-L$sysroot/lib -Wl,--disable-new-dtags,-rpath,$sysroot/lib -static-libstdc++ -static-libgcc"
printf 'int main() { return 0; }\n' > "$HOME/cc.cpp"
$CXX -std=c++17 $LDFLAGS "$HOME/cc.cpp" -o "$HOME/cc" && "$HOME/cc" && echo "compiler ok: $($CXX --version | head -1)"
skip=()
for package in "$@"; do skip+=(--no-install-package "$package"); done
UV_PROJECT_ENVIRONMENT=$prefix uv sync --frozen --no-install-project --managed-python \
  --project "envs/$name" "${skip[@]}"
built=$(UV_PROJECT_ENVIRONMENT=$prefix uv pip list --python "$prefix/bin/python" 2>/dev/null \
  | grep -i -E '^(numpy|pillow|contourpy) ' | tr -s ' ' | tr '\n' ';' || true)
"$prefix/bin/python" -c "import torch, numpy, PIL.Image, contourpy, matplotlib.pyplot; print('imports ok', torch.__version__, numpy.__version__)"
echo "left out: $*; built from sdist on glibc 2.17: $built" > "$prefix/.menagerie-cluster"
echo "$lock" > "$prefix/.menagerie-lock"
echo "built: $lock ($(cat "$prefix/.menagerie-cluster"))"
