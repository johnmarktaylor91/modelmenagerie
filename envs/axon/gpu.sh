# Sourced by a gpu AXON batch (cluster.job_script): the toolchain a CUDA extension builds with
# inside an attempt (nvcc and a host compiler it accepts), for the L40 and L40S nodes (sm_89).
set +u  # lmod reads unset variables
source /etc/profile.d/z00-lmod.sh
module load cuda/12.4.0 gcc/10.4
set -u
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export CC=$(command -v gcc) CXX=$(command -v g++) TORCH_CUDA_ARCH_LIST=8.9
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
