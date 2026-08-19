# syntax=docker/dockerfile:1

# ---- deps ---------------------------------------------------------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- builder ------------------------------------------------------------
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# This app has no backend - every setting is NEXT_PUBLIC_ and Next inlines it
# into the client bundle during `next build`. That means these must be present
# here, at build time; setting them on the running container does nothing.
# Nothing secret lives here (the AAI login is a public OAuth2 client using
# PKCE), so baking them into the image is fine. Defaults mirror .env.example.
ARG NEXT_PUBLIC_WSI_ARCHIVE_BASE_URL="https://cbioportal.wsi-archive.bbmri-eric.eu/wsi-service/v3"
ARG NEXT_PUBLIC_WSI_AAI_AUTH_ENDPOINT="https://login.aai.lifescience-ri.eu/oidc/authorize"
ARG NEXT_PUBLIC_WSI_AAI_TOKEN_ENDPOINT="https://login.aai.lifescience-ri.eu/oidc/token"
ARG NEXT_PUBLIC_WSI_AAI_CLIENT_ID="26f5be84-af80-4480-b764-f78d4ea6a519"
ARG NEXT_PUBLIC_WSI_AAI_REDIRECT_URI="http://localhost:3000/auth/callback"
# Dev-only escape hatch. Leave empty for real deployments - it would end up in
# the client bundle in plaintext.
ARG NEXT_PUBLIC_WSI_DEV_TOKEN=""
# Path prefix to host the app under, e.g. "/ri-scale". Empty = domain root.
# See normaliseBasePath() in next.config.ts. Must match both the reverse proxy
# location and the path in NEXT_PUBLIC_WSI_AAI_REDIRECT_URI.
ARG NEXT_PUBLIC_BASE_PATH=""

ENV NEXT_PUBLIC_BASE_PATH=$NEXT_PUBLIC_BASE_PATH \
    NEXT_PUBLIC_WSI_ARCHIVE_BASE_URL=$NEXT_PUBLIC_WSI_ARCHIVE_BASE_URL \
    NEXT_PUBLIC_WSI_AAI_AUTH_ENDPOINT=$NEXT_PUBLIC_WSI_AAI_AUTH_ENDPOINT \
    NEXT_PUBLIC_WSI_AAI_TOKEN_ENDPOINT=$NEXT_PUBLIC_WSI_AAI_TOKEN_ENDPOINT \
    NEXT_PUBLIC_WSI_AAI_CLIENT_ID=$NEXT_PUBLIC_WSI_AAI_CLIENT_ID \
    NEXT_PUBLIC_WSI_AAI_REDIRECT_URI=$NEXT_PUBLIC_WSI_AAI_REDIRECT_URI \
    NEXT_PUBLIC_WSI_DEV_TOKEN=$NEXT_PUBLIC_WSI_DEV_TOKEN \
    NEXT_TELEMETRY_DISABLED=1

# Needs network access: app/layout.tsx pulls Geist via next/font/google.
RUN npm run build

# ---- runner -------------------------------------------------------------
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY --from=builder --chown=node:node /app/public ./public
# `output: "standalone"` bundles server.js + traced node_modules, but neither
# public/ nor .next/static - both have to come across explicitly.
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static

USER node
EXPOSE 3000
CMD ["node", "server.js"]
