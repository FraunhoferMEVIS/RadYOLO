ARG UV_VERSION="0.5.18"

FROM registry.fme.lan/ghcr/astral-sh/uv:${UV_VERSION} AS uv-base

FROM registry.fme.lan/dockerhub/python:3.12-slim

LABEL org.opencontainers.image.authors="Laurens Müller-Groh <laurens.mueller-groh@mevis.fraunhofer.de>"

COPY --from=uv-base /uv /bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --compile-bytecode --link-mode=copy

ADD src/ /app/

CMD ["/bin/bash"]