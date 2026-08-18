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
    redirectUri: process.env.NEXT_PUBLIC_WSI_AAI_REDIRECT_URI,
  },
  /** Dev-only escape hatch: skips the AAI login flow entirely and uses this
   * token as-is for every archive call. Meant for trying the app out before
   * the redirect URI is registered - see README. Never set this in a real
   * deployment (the token sits in the client bundle in plaintext). */
  devToken: process.env.NEXT_PUBLIC_WSI_DEV_TOKEN,
};
