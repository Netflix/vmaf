**fix(dev):** Replace the broad `COPY . /build/vmaf/` in `dev/Containerfile`
with explicit per-subtree `COPY` directives so that `.corpus/` (up to 781 GB
on developer hosts) is never included in the image layers even when
`docker compose` is invoked without `--project-directory` (which causes
`context: .` to resolve to `dev/` rather than the repo root, bypassing the
root `.dockerignore`). Closes the overlay2 accumulation regression.
