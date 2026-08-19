import { normaliseBasePath } from "./basePath";

// All client-side (this app has no backend of its own left) - NEXT_PUBLIC_
// prefix is required for Next.js to inline these into the browser bundle.
// Nothing secret lives here: the AAI login is a public OAuth2 client using
// PKCE, so there is no client_secret to protect.
export const config = {
  wsiArchive: {
    baseUrl: process.env.NEXT_PUBLIC_WSI_ARCHIVE_BASE_URL,
  },
  aai: {
    authEndpoint: process.env.NEXT_PUBLIC_WSI_AAI_AUTH_ENDPOINT,
    tokenEndpoint: process.env.NEXT_PUBLIC_WSI_AAI_TOKEN_ENDPOINT,
    clientId: process.env.NEXT_PUBLIC_WSI_AAI_CLIENT_ID,
    /** Normally unset - see resolveRedirectUri(), which derives it from
     * wherever the app is actually being served. Only set this to pin the
     * value to something the origin can't tell you. */
    redirectUriOverride: process.env.NEXT_PUBLIC_WSI_AAI_REDIRECT_URI,
  },
  /** Path prefix the app is mounted under, e.g. "/ri-scale". Must match the
   * basePath baked in at build time - both read the same variable. */
  basePath: normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH) ?? "",
  /** Dev-only escape hatch: skips the AAI login flow entirely and uses this
   * token as-is for every archive call. Meant for trying the app out before
   * the redirect URI is registered - see README. Never set this in a real
   * deployment (the token sits in the client bundle in plaintext). */
  devToken: process.env.NEXT_PUBLIC_WSI_DEV_TOKEN,
};

/** The OAuth2 redirect_uri to hand to AAI.
 *
 * Derived from the origin the app is currently served from, rather than baked
 * in at build time - one image then works on localhost, staging and prod
 * without a rebuild, and can't drift out of sync with the domain it's actually
 * running on. Deriving it is not a hole: AAI only honours redirect URIs
 * pre-registered for the client, so pointing this at some other origin just
 * gets the login rejected.
 *
 * Returns undefined during prerender, where there is no origin to read. Both
 * callers (startLogin, handleCallback) run in the browser, so that never
 * surfaces in practice. */
export function resolveRedirectUri(): string | undefined {
  if (config.aai.redirectUriOverride) return config.aai.redirectUriOverride;
  if (typeof window === "undefined") return undefined;
  return `${window.location.origin}${config.basePath}/auth/callback`;
}
