# syntax=docker/dockerfile:1
# Export the effective (dockerignore-filtered) inputs for pre-publication review.
FROM scratch AS build-context
COPY . /

# ── Stage 1: build the React UI ───────────────────────────────────
FROM node:20-slim AS ui-builder
ARG CODEX_VERSION=0.145.0
WORKDIR /ui
COPY ui/mira/package.json ui/mira/package-lock.json ./
RUN npm ci --no-audit --no-fund
RUN npm install --global --no-audit --no-fund "@openai/codex@${CODEX_VERSION}"
COPY ui/mira/ ./
RUN npm run build

# Build only the selected Python package; never install the repository in runtime.
FROM python:3.12-slim AS python-builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/mira/ src/mira/
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[serve,bedrock]"

# ── Stage 3: backend + bundled UI ─────────────────────────────────
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/miracodeai/mira"
LABEL org.opencontainers.image.description="Self-hostable AI code reviewer"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
# Mount wheels so neither build inputs nor wheel archives persist in a layer.
RUN --mount=type=bind,from=python-builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels "mira-reviewer[serve,bedrock]"

# Codex CLI is bundled for the optional codex-cli backend. Keep the version
# pinned above so image rebuilds do not silently change the review runtime.
COPY --from=ui-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=ui-builder /usr/local/lib/node_modules/@openai/codex /usr/local/lib/node_modules/@openai/codex
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex

# Pull the built UI in from stage 1. webhooks.create_app() picks this up
# automatically and serves it at / with SPA fallback.
COPY --from=ui-builder /ui/dist /app/ui_dist

EXPOSE 8000
# ENTRYPOINT (not CMD) so `docker run … image --config /app/mira.yaml`
# appends the args to `mira serve` instead of replacing the command.
ENTRYPOINT ["mira", "serve"]
