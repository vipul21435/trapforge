# syntax=docker/dockerfile:1
# TrapForge CLI image: uv builds the virtualenv, the runtime stage keeps only the venv and
# the bundled examples, and everything runs as an unprivileged user.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, so editing the source does not invalidate this layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL project=trapforge \
      org.opencontainers.image.title="trapforge" \
      org.opencontainers.image.source="https://github.com/vipul21435/trapforge" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin forge
COPY --from=build /opt/venv /opt/venv
COPY --chown=forge:forge examples /home/forge/examples
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER forge
WORKDIR /home/forge
ENTRYPOINT ["trapforge"]
CMD ["--help"]
