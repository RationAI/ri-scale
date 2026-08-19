import { useEffect, useState } from "react";
import { config, resolveRedirectUri } from "@/lib/config";
import { generateCodeChallenge, generateCodeVerifier, generateState } from "./pkce";

const STORAGE_KEY = "wsi_auth_tokens";
const PKCE_STORAGE_KEY = "wsi_auth_pkce";
const EXPIRY_BUFFER_MS = 30_000;

interface StoredTokens {
  accessToken: string;
  refreshToken?: string;
  expiresAt: number;
}

function readTokens(): StoredTokens | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredTokens;
  } catch {
    return null;
  }
}

function writeTokens(tokens: StoredTokens) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tokens));
}

export function clearTokens() {
  localStorage.removeItem(STORAGE_KEY);
}

export function isLoggedIn(): boolean {
  if (config.devToken) return true;
  return readTokens() !== null;
}

function requireAaiConfig() {
  const { authEndpoint, tokenEndpoint, clientId } = config.aai;
  // Derived from window.location, so this is browser-only - which both
  // callers below are.
  const redirectUri = resolveRedirectUri();
  if (!authEndpoint || !tokenEndpoint || !clientId || !redirectUri) {
    throw new Error("NEXT_PUBLIC_WSI_AAI_* environment variables must be set");
  }
  return { authEndpoint, tokenEndpoint, clientId, redirectUri };
}

/** Kicks off the OAuth2 Authorization Code + PKCE flow against Life Science
 * AAI. Public client (no client_secret) - nothing secret is held anywhere,
 * which is what makes it safe to run entirely in the browser.
 */
export async function startLogin(): Promise<void> {
  const { authEndpoint, clientId, redirectUri } = requireAaiConfig();
  const verifier = generateCodeVerifier();
  const challenge = await generateCodeChallenge(verifier);
  const state = generateState();

  // Only needed transiently across the redirect round-trip.
  sessionStorage.setItem(PKCE_STORAGE_KEY, JSON.stringify({ verifier, state }));

  const url = new URL(authEndpoint);
  url.searchParams.set("response_type", "code");
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", redirectUri);
  url.searchParams.set(
    "scope",
    "openid email eduperson_entitlement offline_access profile"
  );
  url.searchParams.set("code_challenge", challenge);
  url.searchParams.set("code_challenge_method", "S256");
  url.searchParams.set("state", state);

  window.location.href = url.toString();
}

export async function handleCallback(
  searchParams: URLSearchParams
): Promise<void> {
  const { tokenEndpoint, clientId, redirectUri } = requireAaiConfig();

  const error = searchParams.get("error");
  if (error) {
    throw new Error(
      `AAI login failed: ${error} - ${searchParams.get("error_description") ?? ""}`
    );
  }

  const code = searchParams.get("code");
  const state = searchParams.get("state");
  if (!code || !state) {
    throw new Error("Missing code/state in AAI callback");
  }

  const stashed = sessionStorage.getItem(PKCE_STORAGE_KEY);
  sessionStorage.removeItem(PKCE_STORAGE_KEY);
  if (!stashed) {
    throw new Error(
      "Missing stashed PKCE verifier - did you reload the callback page?"
    );
  }
  const { verifier, state: expectedState } = JSON.parse(stashed) as {
    verifier: string;
    state: string;
  };
  if (state !== expectedState) {
    throw new Error("State mismatch - possible CSRF, aborting login");
  }

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    code,
    redirect_uri: redirectUri,
    client_id: clientId,
    code_verifier: verifier,
  });

  const res = await fetch(tokenEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Token exchange failed (${res.status}): ${text}`);
  }
  const data = (await res.json()) as {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
  };

  writeTokens({
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + data.expires_in * 1000,
  });
}

let refreshInFlight: Promise<string> | null = null;

async function refreshAccessToken(): Promise<string> {
  const { tokenEndpoint, clientId } = requireAaiConfig();
  const tokens = readTokens();
  if (!tokens?.refreshToken) {
    throw new Error("Not logged in (no refresh token available)");
  }

  const body = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: tokens.refreshToken,
    client_id: clientId,
  });

  const res = await fetch(tokenEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    // The refresh token is dead too - only a fresh interactive login can fix
    // this, so drop the stale tokens rather than keep retrying with them.
    clearTokens();
    const text = await res.text().catch(() => "");
    throw new Error(`Token refresh failed (${res.status}): ${text}`);
  }
  const data = (await res.json()) as {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
  };

  writeTokens({
    accessToken: data.access_token,
    refreshToken: data.refresh_token ?? tokens.refreshToken,
    expiresAt: Date.now() + data.expires_in * 1000,
  });
  return data.access_token;
}

export async function getValidAccessToken(): Promise<string> {
  if (config.devToken) return config.devToken;
  const tokens = readTokens();
  if (tokens && tokens.expiresAt - EXPIRY_BUFFER_MS > Date.now()) {
    return tokens.accessToken;
  }
  return forceRefreshAccessToken();
}

/** Forces a fresh token even if the cached one looked unexpired - used after
 * the archive itself rejects our token with a 401. */
export async function forceRefreshAccessToken(): Promise<string> {
  if (config.devToken) return config.devToken;
  if (!refreshInFlight) {
    refreshInFlight = refreshAccessToken().finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

/**
 * Deliberately NOT useSyncExternalStore here, despite that being the more
 * idiomatic React pattern for external state like localStorage: it renders
 * the server snapshot (always `false`, since the server never has a token)
 * on the first pass, then corrects to the real client value on a later
 * render. That transient `false` is indistinguishable from "genuinely
 * logged out" to a consumer, so a redirect-on-logged-out effect (see the
 * browse page) fires immediately on the transient value and navigates away
 * *before* the correction lands - confirmed by hand, this actually happens.
 * The three-state model below (unknown / logged-out / logged-in) avoids
 * that: consumers must check `ready` before treating `loggedIn` as real.
 */
export function useAuth(): { loggedIn: boolean; ready: boolean } {
  const [loggedIn, setLoggedIn] = useState(false);
  const [ready, setReady] = useState(false);

  /* eslint-disable react-hooks/set-state-in-effect -- intentional: see the
     function doc comment above for why this can't be useSyncExternalStore */
  useEffect(() => {
    setLoggedIn(isLoggedIn());
    setReady(true);
    /* eslint-enable react-hooks/set-state-in-effect */

    const onStorage = () => setLoggedIn(isLoggedIn());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return { loggedIn, ready };
}
