# envs/axon

Not an environment: the pieces an AXON batch (`menagerie/pipeline/cluster.py`) adds on the
cluster. A batch builds the recipe's own environment from its committed lock under `envs/`.

- `uv.txt`: the pinned static uv release and its sha256 (`cluster._uv_binary` checks it).
- `build.sh`: builds a committed lock on the cluster's glibc 2.17 (what differs is written
  to the prefix's `.menagerie-cluster`).
- `gpu.sh`: the CUDA toolchain a gpu batch sources so compiled extensions can build.
