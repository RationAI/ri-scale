import { config } from "@/lib/config";
import { forceRefreshAccessToken, getValidAccessToken } from "@/lib/auth/client";

function requireBaseUrl(): string {
  const { baseUrl } = config.wsiArchive;
  if (!baseUrl) throw new Error("NEXT_PUBLIC_WSI_ARCHIVE_BASE_URL must be set");
  return baseUrl;
}

/** Authenticated fetch straight from the browser to the remote WSI archive
 * (its CORS policy allows any origin plus the Authorization header - no
 * server proxy needed). `path` is relative to the base URL; leading slashes
 * are stripped so callers can't accidentally discard the "/v3" prefix.
 *
 * Retries once, with a forced token refresh, on a 401.
 */
export async function wsiFetch(
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  const url = `${requireBaseUrl().replace(/\/$/, "")}/${path.replace(/^\/+/, "")}`;
  const doFetch = async (token: string) =>
    fetch(url, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${token}` },
    });

  const token = await getValidAccessToken();
  const res = await doFetch(token);
  if (res.status !== 401) return res;

  const refreshed = await forceRefreshAccessToken();
  return doFetch(refreshed);
}
