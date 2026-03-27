# Caching and Docker Log Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ccache + pip caching to the GitHub Actions build job and fix Docker log buffering so container output streams in real time to the GHA console.

**Architecture:** All changes are confined to `.github/workflows/build.yml` and `scripts/build_target.sh`. The GHA runner host mounts two cache directories (`~/.cache/ccache` and `~/.cache/pip`) into the Docker container via `-v` bind mounts. Inside the container, ccache is installed and `CC`/`CXX` are set to `ccache gcc`/`ccache g++`. Log streaming is fixed by setting `PYTHONUNBUFFERED=1` and using `stdbuf -oL` on the bash shell that runs the build script.

**Tech Stack:** GitHub Actions (`actions/cache@v4`), ccache, Docker bind mounts, bash `stdbuf`.

---

### Task 1: Fix Docker log streaming

The container currently buffers stdout/stderr, meaning GHA shows nothing until the container exits. Fix: pass `-e PYTHONUNBUFFERED=1` (Python pip/huggingface-cli output) and wrap the entrypoint with `stdbuf -oL` so bash also runs line-buffered. Also add `--progress=plain` on the `docker build` step if one is ever added in future — not needed here since there is no build step, only `docker run`.

**Files:**
- Modify: `.github/workflows/build.yml:56-74`

- [ ] **Step 1: Add unbuffering env vars and stdbuf to the docker run command**

In `.github/workflows/build.yml`, replace the `docker run` block (lines 58–74) with:

```yaml
          docker run --rm \
            -v "$(pwd)/scripts:/scripts:ro" \
            -v "$(pwd)/builds:/manifest:ro" \
            -v "$(pwd)/output:/output" \
            -e TARGET_ID \
            -e ORT_VERSION \
            -e HF_REPO_ID \
            -e HF_REVISION \
            -e HF_PRIMARY \
            -e HF_COMPANIONS \
            -e CPU_TUNING \
            -e EXECUTION_PROVIDER \
            -e MINIMAL_BUILD \
            -e HF_TOKEN \
            -e OUTPUT_DIR=/output \
            -e PYTHONUNBUFFERED=1 \
            "${{ matrix.target.container_image }}" \
            stdbuf -oL /bin/bash /scripts/build_target.sh
```

Key changes:
- `-e PYTHONUNBUFFERED=1` — prevents Python from buffering stdout (affects huggingface-cli, pip, gen_opkernel_def.py)
- `stdbuf -oL /bin/bash /scripts/build_target.sh` — replaces `/scripts/build_target.sh`; `stdbuf -oL` sets line-buffered stdout for the bash process and all child processes that inherit the fd

- [ ] **Step 2: Verify the change is syntactically valid YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "fix: stream docker container logs to GHA console in real time"
```

---

### Task 2: Add ccache and pip cache directories to the Dockerfile

ccache must be installed in the container image. The pip cache directory must also be available. Install `ccache` via `dnf` in the Dockerfile.

**Files:**
- Modify: `docker/lambda-build.Dockerfile:6-18`

- [ ] **Step 1: Add ccache to dnf install block**

In `docker/lambda-build.Dockerfile`, add `ccache \` to the `dnf install` list:

```dockerfile
RUN dnf install -y \
        cmake \
        ninja-build \
        git \
        gcc \
        gcc-c++ \
        ccache \
        jq \
        python3 \
        python3-pip \
        tar \
        gzip \
        which \
    && dnf clean all
```

- [ ] **Step 2: Verify Dockerfile syntax by doing a dry-run parse**

```bash
docker build --no-cache --dry-run -f docker/lambda-build.Dockerfile docker/ 2>&1 || true
```

If `--dry-run` is not supported by your Docker version, just verify the file looks correct:

```bash
python3 -c "
import subprocess, sys
r = subprocess.run(['docker', 'build', '--help'], capture_output=True)
print('docker available:', r.returncode == 0)
"
```

Expected: no syntax errors visible in the file.

- [ ] **Step 3: Commit**

```bash
git add docker/lambda-build.Dockerfile
git commit -m "feat: install ccache in Lambda build container"
```

---

### Task 3: Mount cache volumes and configure ccache in the workflow

The GHA runner host caches `~/.cache/ccache` and `~/.cache/pip` between runs using `actions/cache@v4`. Both directories are bind-mounted into the Docker container. Inside the container, `CC=ccache gcc` and `CXX=ccache g++` are set via `-e` so ORT's CMake picks them up.

Cache key strategy:
- **ccache key:** `ccache-arm64-ort-<ort_version>-<target_id>` — scoped per ORT version + target so different versions don't share corrupt object files. Restore key falls back to `ccache-arm64-ort-<ort_version>-` to reuse partial caches from prior runs of the same ORT version.
- **pip key:** `pip-arm64-<hash of Dockerfile>` — invalidates when the Dockerfile's pip install block changes.

**Files:**
- Modify: `.github/workflows/build.yml:41-81`

- [ ] **Step 1: Add cache steps before the build step**

In the `build` job, insert two new steps between `uses: actions/checkout@v4` and `Build target in Lambda container`:

```yaml
      - name: Restore ccache
        uses: actions/cache@v4
        with:
          path: ~/.cache/ccache
          key: ccache-arm64-ort-${{ matrix.target.ort_version }}-${{ matrix.target.target_id }}-${{ github.sha }}
          restore-keys: |
            ccache-arm64-ort-${{ matrix.target.ort_version }}-${{ matrix.target.target_id }}-
            ccache-arm64-ort-${{ matrix.target.ort_version }}-

      - name: Restore pip cache
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-arm64-${{ hashFiles('docker/lambda-build.Dockerfile') }}
          restore-keys: |
            pip-arm64-
```

- [ ] **Step 2: Add cache volume mounts and ccache env vars to docker run**

Extend the `docker run` command (already modified in Task 1) with:
- `-v "$HOME/.cache/ccache:/root/.cache/ccache"` bind mount
- `-v "$HOME/.cache/pip:/root/.cache/pip"` bind mount
- `-e CC=ccache gcc` — tells CMake's C compiler to use ccache
- `-e CXX=ccache g++` — tells CMake's C++ compiler to use ccache
- `-e CCACHE_DIR=/root/.cache/ccache` — explicit ccache dir inside container

The full updated `docker run` block becomes:

```yaml
          mkdir -p ~/.cache/ccache ~/.cache/pip output
          docker run --rm \
            -v "$(pwd)/scripts:/scripts:ro" \
            -v "$(pwd)/builds:/manifest:ro" \
            -v "$(pwd)/output:/output" \
            -v "$HOME/.cache/ccache:/root/.cache/ccache" \
            -v "$HOME/.cache/pip:/root/.cache/pip" \
            -e TARGET_ID \
            -e ORT_VERSION \
            -e HF_REPO_ID \
            -e HF_REVISION \
            -e HF_PRIMARY \
            -e HF_COMPANIONS \
            -e CPU_TUNING \
            -e EXECUTION_PROVIDER \
            -e MINIMAL_BUILD \
            -e HF_TOKEN \
            -e OUTPUT_DIR=/output \
            -e PYTHONUNBUFFERED=1 \
            -e CC="ccache gcc" \
            -e CXX="ccache g++" \
            -e CCACHE_DIR=/root/.cache/ccache \
            "${{ matrix.target.container_image }}" \
            stdbuf -oL /bin/bash /scripts/build_target.sh
```

Note: `mkdir -p ~/.cache/ccache ~/.cache/pip` ensures the dirs exist on the runner before mounting (Docker will create them as root-owned dirs otherwise, which breaks the cache save step).

- [ ] **Step 3: Add a ccache stats step after the build step for visibility**

After the `Build target in Lambda container` step, add:

```yaml
      - name: Show ccache stats
        if: always()
        run: docker run --rm -v "$HOME/.cache/ccache:/root/.cache/ccache" -e CCACHE_DIR=/root/.cache/ccache "${{ matrix.target.container_image }}" ccache --show-stats
```

`if: always()` ensures stats are shown even if the build fails — useful for debugging cache hit rates.

- [ ] **Step 4: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "feat: add ccache and pip caching to GHA build job"
```

---

### Task 4: Pass ccache compiler wrappers through to ORT's build.py

ORT's `build.py` invokes CMake which invokes the compiler. For ccache to intercept compilation, `CC` and `CXX` env vars must be set in the environment that `build.py` runs in. Since they are already set via Docker `-e CC=...` and `-e CXX=...`, CMake will pick them up automatically via `$ENV{CC}` / `$ENV{CXX}`.

However, there is one subtlety: `CC=ccache gcc` (with a space) is treated as a single string by the shell env, not a two-word command. CMake handles this correctly when `CC` contains a path or a wrapper. But to be safe and explicit, it is better to use a wrapper script or `ccache` as the compiler path via `CMAKE_C_COMPILER_LAUNCHER`.

The cleaner approach is to pass `CMAKE_C_COMPILER_LAUNCHER=ccache` and `CMAKE_CXX_COMPILER_LAUNCHER=ccache` as cmake_extra_defines instead of setting `CC`/`CXX`. This is the CMake-endorsed way to use ccache.

**Files:**
- Modify: `scripts/build_target.sh:109-130`
- Modify: `.github/workflows/build.yml` (remove `CC`/`CXX` env vars, they are no longer needed)

- [ ] **Step 1: Replace CC/CXX env vars with CMAKE_*_COMPILER_LAUNCHER in build_target.sh**

In `scripts/build_target.sh`, update the `CMAKE_EXTRA_DEFINES` block:

```bash
CMAKE_EXTRA_DEFINES=(
    "onnxruntime_BUILD_SHARED_LIB=ON"
    "CMAKE_C_COMPILER_LAUNCHER=ccache"
    "CMAKE_CXX_COMPILER_LAUNCHER=ccache"
)
```

These are passed to `build.py --cmake_extra_defines` which prepends `-D` internally, so CMake sees:
- `-DCMAKE_C_COMPILER_LAUNCHER=ccache`
- `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`

- [ ] **Step 2: Remove CC and CXX from the docker run env block in the workflow**

In `.github/workflows/build.yml`, remove these two lines from the `docker run` command:

```yaml
            -e CC="ccache gcc" \
            -e CXX="ccache g++" \
```

`CCACHE_DIR` stays — ccache needs to know where its cache lives.

- [ ] **Step 3: Verify YAML is valid and build script looks correct**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))" && echo "YAML OK"
python3 -m pytest tests/ -v
```

Expected: `YAML OK` and `14 passed`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_target.sh .github/workflows/build.yml
git commit -m "fix: use CMAKE_C_COMPILER_LAUNCHER instead of CC/CXX for ccache integration"
```

---

## Self-Review

**Spec coverage:**
- [x] Docker log streaming → Task 1 (`stdbuf -oL`, `PYTHONUNBUFFERED=1`)
- [x] pip caching → Task 3 (`actions/cache` + `-v` mount for `~/.cache/pip`)
- [x] ccache → Tasks 2, 3, 4 (install, mount, `CMAKE_C_COMPILER_LAUNCHER`)

**Placeholder scan:** None found.

**Type consistency:** No types; all shell/YAML. Env var names `CCACHE_DIR`, `PYTHONUNBUFFERED` are consistent across tasks.

**Ordering:** Tasks 1→2→3→4 are sequential: streaming fix first (independent), then Dockerfile, then workflow caching wiring, then the ccache/CMake integration correction. Task 4 corrects a subtlety introduced in Task 3 (CC/CXX approach → compiler launcher approach).
