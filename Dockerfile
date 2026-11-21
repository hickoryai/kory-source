# syntax=docker/dockerfile:1
# Base images. A release passes each one with its resolved digest
# (scripts/resolve_reviewer_deps.py), so the image records exactly what it
# was built from. The major lines are chosen here and change only by review.
ARG NODE_UI_IMAGE=node:20-slim
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.12-slim
ARG RUST_IMAGE=rust:1-slim-trixie

# Export the effective (dockerignore-filtered) inputs for pre-publication review.
FROM scratch AS build-context
COPY . /

# ── Reviewer sources ──────────────────────────────────────────────
# Source trees of the two reviewer programs: the ACP session router and the
# agent CLI. Both are empty here, so a plain `docker build .` produces an
# image without them. scripts/build_reviewer_image.sh checks each out at its
# newest main commit and passes the trees as the named build contexts
# `router-acp-src` and `agent-cli-src`. No repository address is recorded in
# this file, a build argument, or the image history.
FROM scratch AS router-acp-src
FROM scratch AS agent-cli-src

# ── Stage 1: build the React UI ───────────────────────────────────
FROM ${NODE_UI_IMAGE} AS ui-builder
WORKDIR /ui
COPY ui/mira/package.json ui/mira/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/mira/ ./
RUN npm run build

# Build only the selected Python package; never install the repository in runtime.
FROM ${PYTHON_IMAGE} AS python-builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/mira/ src/mira/
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[serve,bedrock]"

# ── Reviewer programs, built from source ──────────────────────────
# Each build reads its source from the named context and leaves its binaries
# in /out. An empty context leaves /out empty. The toolchain a source tree
# pins (rust-toolchain.toml) is installed by rustup on first use.
FROM ${RUST_IMAGE} AS reviewer-rust
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang cmake git libclang-dev libdbus-1-dev \
        libprotobuf-dev libssl-dev libxcb1-dev pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM reviewer-rust AS router-acp-build
RUN --mount=type=bind,from=router-acp-src,target=/src,rw \
    --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/target,id=router-acp-target,sharing=locked \
    mkdir -p /out \
    && if [ -f /src/Cargo.toml ]; then \
         cd /src \
         && CARGO_TARGET_DIR=/target cargo build --locked --release \
              --bin router-acp --bin router-resource-run \
         && install -m 0755 /target/release/router-acp /target/release/router-resource-run /out/; \
       fi

FROM reviewer-rust AS agent-cli-build
RUN --mount=type=bind,from=agent-cli-src,target=/src,rw \
    --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/target,id=agent-cli-target,sharing=locked \
    mkdir -p /out \
    && if [ -f /src/Cargo.toml ]; then \
         cd /src \
         && CARGO_TARGET_DIR=/target cargo build --locked --release -p goose-cli --bin kory \
         && install -m 0755 /target/release/kory /out/; \
       fi

# ── ACP agents ────────────────────────────────────────────────────
# The Codex CLI serves the optional codex-cli backend and is always installed.
# REVIEWER_AGENTS=1 adds the agents the router's presets launch: the Claude
# and Codex ACP adapters and the Grok CLI. Versions are resolved per build;
# `latest` is the newest published version at build time. npm runs no install
# scripts for global packages, so each package's postinstall (where one places
# a native binary) is run here. Every command must answer `--version`, so a
# missing binary fails this build and not a review.
FROM ${NODE_IMAGE} AS agents
ARG REVIEWER_AGENTS=0
ARG CODEX_VERSION=latest
ARG CLAUDE_AGENT_ACP_VERSION=latest
ARG CODEX_ACP_VERSION=latest
ARG GROK_VERSION=latest
RUN set -eu; \
    packages="@openai/codex@${CODEX_VERSION}"; \
    names="@openai/codex"; \
    commands="codex"; \
    if [ "${REVIEWER_AGENTS}" = "1" ]; then \
      for version in "${CODEX_VERSION}" "${CLAUDE_AGENT_ACP_VERSION}" "${CODEX_ACP_VERSION}" "${GROK_VERSION}"; do \
        if [ "${version}" = "latest" ]; then \
          echo "REVIEWER_AGENTS=1 needs resolved versions; run scripts/build_reviewer_image.sh" >&2; exit 1; \
        fi; \
      done; \
      packages="${packages} @agentclientprotocol/claude-agent-acp@${CLAUDE_AGENT_ACP_VERSION}"; \
      packages="${packages} @agentclientprotocol/codex-acp@${CODEX_ACP_VERSION} @xai-official/grok@${GROK_VERSION}"; \
      names="${names} @agentclientprotocol/claude-agent-acp @agentclientprotocol/codex-acp @xai-official/grok"; \
      commands="${commands} claude-agent-acp codex-acp grok"; \
    fi; \
    npm install --global --no-audit --no-fund ${packages}; \
    mkdir -p /agents/bin /agents/versions; \
    for name in ${names}; do \
      dir="$(npm root -g)/${name}"; \
      script="$(node -p "(require('${dir}/package.json').scripts||{}).postinstall||''")"; \
      if [ -n "${script}" ]; then (cd "${dir}" && sh -c "${script}"); fi; \
      node -p "require('${dir}/package.json').version" > "/agents/versions/$(basename "${name}")"; \
    done; \
    for command in ${commands}; do \
      "${command}" --version; \
      cp -P "/usr/local/bin/${command}" /agents/bin/; \
    done; \
    rm -rf /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/corepack /root/.npm

# ── Stage 3: backend + bundled UI ─────────────────────────────────
FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/miracodeai/mira"
LABEL org.opencontainers.image.description="Self-hostable AI code reviewer"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
# git serves repository reads. The three libraries are what the agent CLI
# links against at run time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libdbus-1-3 libgomp1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
# Mount wheels so neither build inputs nor wheel archives persist in a layer.
RUN --mount=type=bind,from=python-builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels "mira-reviewer[serve,bedrock]"

# Node and the agent packages. The command links are relative, so they
# resolve under /usr/local here as they did in the agents stage.
COPY --from=agents /usr/local/bin/node /usr/local/bin/node
COPY --from=agents /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=agents /agents/bin/ /usr/local/bin/
COPY --from=agents /agents/versions/ /etc/kory-source/agent-versions/

# The reviewer programs. Both copies are empty in a plain build.
COPY --from=router-acp-build /out/ /usr/local/bin/
COPY --from=agent-cli-build /out/ /usr/local/bin/

# What this image was built from. A release passes every value; a plain
# build records `unset` for the reviewer programs. The file is what
# `python -m mira.auto_review.reviewer_info` prints.
ARG REVIEWER_AGENTS=0
ARG ROUTER_ACP_REVISION=unset
ARG AGENT_CLI_REVISION=unset
ARG NODE_IMAGE
ARG PYTHON_IMAGE
LABEL io.kory-source.dep.router-acp="${ROUTER_ACP_REVISION}"
LABEL io.kory-source.dep.agent-cli="${AGENT_CLI_REVISION}"
LABEL io.kory-source.dep.node-image="${NODE_IMAGE}"
LABEL io.kory-source.dep.python-image="${PYTHON_IMAGE}"
# The router's Grok preset starts exactly ~/.grok/bin/grok. The agent CLI's
# own settings name the router as its one provider, with no telemetry.
# With REVIEWER_AGENTS=1 every reviewer command must be present and answer.
RUN set -eu; \
    printf '{"router_acp": "%s", "agent_cli": "%s", "node_image": "%s", "python_image": "%s"}\n' \
      "${ROUTER_ACP_REVISION}" "${AGENT_CLI_REVISION}" "${NODE_IMAGE}" "${PYTHON_IMAGE}" \
      > /etc/kory-source/reviewer-build.json; \
    if [ -e /usr/local/bin/grok ]; then \
      mkdir -p /root/.grok/bin && ln -s /usr/local/bin/grok /root/.grok/bin/grok; \
    fi; \
    if [ -x /usr/local/bin/kory ]; then \
      mkdir -p /root/.config/goose; \
      printf '%s\n' 'GOOSE_TELEMETRY_ENABLED: false' 'GOOSE_MODE: auto' 'active_provider: router-acp' \
        'providers:' '  router-acp:' '    enabled: true' '    model: default' '    configured: true' \
        > /root/.config/goose/config.yaml; \
    fi; \
    codex --version; \
    if [ "${REVIEWER_AGENTS}" = "1" ]; then \
      test -x /usr/local/bin/router-resource-run; \
      router-acp --version; \
      kory --version; \
      claude-agent-acp --version; \
      codex-acp --version; \
      /root/.grok/bin/grok --version; \
      python -m mira.auto_review.reviewer_info --check; \
    fi
# The Claude agent runs with the preset's automatic mode as root only when it
# is told this is a sandbox. The router configuration and its logins are
# mounted from another host, so this one never rotates them.
ENV IS_SANDBOX=1 \
    ROUTER_ACP_CREDENTIAL_REFRESH=off

# Pull the built UI in from stage 1. webhooks.create_app() picks this up
# automatically and serves it at / with SPA fallback.
COPY --from=ui-builder /ui/dist /app/ui_dist

EXPOSE 8000
# ENTRYPOINT (not CMD) so `docker run … image --config /app/mira.yaml`
# appends the args to `mira serve` instead of replacing the command.
ENTRYPOINT ["mira", "serve"]
