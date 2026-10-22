# TODO

## Refresh hashes using a fresh output file

`uv pip compile` reuses hashes from an existing output file. The commands below
follow the [uv maintainer's workaround](https://github.com/astral-sh/uv/issues/16104):
they constrain compilation with the existing requirements file, pass `--refresh`,
and write to a new temporary output file. This keeps the existing version pins
while retrieving fresh package metadata and hashes from the Red Hat index. Each
original file is replaced only after its compile command succeeds; no manual hash
deletion is needed.

Complete both regeneration sections, review the diffs, and run deduplication.
The deduplicated CPU files omit some shared packages, so review any versions
newly resolved for those packages as well.

## Regenerate CPU requirements for RHOAI 3.6

Run these commands from the repository root. The project supports Python 3.11 and 3.12; use Python 3.12 because the generated requirement files target Python 3.12.

```bash
cd /Users/gnaulak/workspace/worktrees/downstream/lm-evaluation-harness

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

python --version

export CPU_INDEX="https://packages.redhat.com/api/pypi/public-rhai/rhoai/3.6/cpu-ubi9-test/simple/"
CPU_OUTPUT_DIR="$(mktemp -d)"

uv pip compile pyproject.toml \
  --extra api \
  --extra ifeval \
  --extra math \
  --extra ibm_watsonx_ai \
  --python-platform linux \
  --python-version 3.12 \
  --no-emit-package lm_eval \
  --index-strategy first-index \
  --emit-index-url \
  --emit-index-annotation \
  --generate-hashes \
  --prerelease=if-necessary \
  --no-strip-markers \
  --default-index "$CPU_INDEX" \
  --refresh \
  --constraint requirements/constraints.txt \
  --constraint requirements/cpu/requirements.txt \
  -o "$CPU_OUTPUT_DIR/requirements.txt" && \
  mv "$CPU_OUTPUT_DIR/requirements.txt" requirements/cpu/requirements.txt

uv pip compile requirements/requirements-build.in \
  --python-platform linux \
  --python-version 3.12 \
  --index-strategy first-index \
  --emit-index-url \
  --emit-index-annotation \
  --generate-hashes \
  --default-index "$CPU_INDEX" \
  --refresh \
  --constraint requirements/cpu/requirements-build.txt \
  -o "$CPU_OUTPUT_DIR/requirements-build.txt" && \
  mv "$CPU_OUTPUT_DIR/requirements-build.txt" requirements/cpu/requirements-build.txt
```

Use the configured Red Hat package credentials; do not embed credentials in the index URL.

Verify the generated files:

```bash
rg -n '3\.6(?:-EA[12])?' requirements/cpu
git diff --check
git diff -- requirements/cpu
```

Confirm that the files reference `3.6`. Review any package-version or hash changes before committing.

## Regenerate CUDA requirements for RHOAI 3.6

Run these commands from the repository root. Use Python 3.12 because the generated requirement files target Python 3.12.

```bash
cd /Users/gnaulak/workspace/worktrees/downstream/lm-evaluation-harness

source .venv/bin/activate
python --version

export CUDA_INDEX="https://packages.redhat.com/api/pypi/public-rhai/rhoai/3.6/cuda13.0-ubi9-test/simple/"
CUDA_OUTPUT_DIR="$(mktemp -d)"

uv pip compile pyproject.toml \
  --extra api \
  --extra ifeval \
  --extra math \
  --extra ibm_watsonx_ai \
  --python-platform linux \
  --python-version 3.12 \
  --no-emit-package lm_eval \
  --index-strategy first-index \
  --emit-index-url \
  --emit-index-annotation \
  --generate-hashes \
  --prerelease=if-necessary \
  --no-strip-markers \
  --default-index "$CUDA_INDEX" \
  --refresh \
  --constraint requirements/constraints.txt \
  --constraint requirements/cuda/requirements.txt \
  -o "$CUDA_OUTPUT_DIR/requirements.txt" && \
  mv "$CUDA_OUTPUT_DIR/requirements.txt" requirements/cuda/requirements.txt

uv pip compile requirements/requirements-build.in \
  --python-platform linux \
  --python-version 3.12 \
  --index-strategy first-index \
  --emit-index-url \
  --emit-index-annotation \
  --generate-hashes \
  --default-index "$CUDA_INDEX" \
  --refresh \
  --constraint requirements/cuda/requirements-build.txt \
  -o "$CUDA_OUTPUT_DIR/requirements-build.txt" && \
  mv "$CUDA_OUTPUT_DIR/requirements-build.txt" requirements/cuda/requirements-build.txt
```

Use the configured Red Hat package credentials; do not embed credentials in the index URL.

Verify the generated files:

```bash
rg -n '3\.6(?:-EA[12])?' requirements/cuda
git diff --check
git diff -- requirements/cuda
```

Confirm that the files reference `3.6`. Review any package-version or hash changes before committing.

## dedup

```bash
python scripts/dedup_prefetch_requirements.py \
    --primary requirements/cuda \
    --secondary requirements/cpu
```

## Downstream PR

- After raising the downstream PR, post `/build-konflux` as a PR comment to trigger the Konflux build.
