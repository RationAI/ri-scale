/** Shared by next.config.ts (which feeds it to Next's `basePath`) and the
 * client config, so both agree on what NEXT_PUBLIC_BASE_PATH means.
 *
 * Returns undefined rather than "" for "no prefix": Next rejects an empty
 * string for `basePath`, it has to be absent entirely. Callers wanting a
 * concatenable value should use `?? ""`. */
export function normaliseBasePath(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim().replace(/\/+$/, "");
  if (!trimmed) return undefined;
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}
