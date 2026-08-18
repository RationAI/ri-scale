"use client";

import { useEffect, useState } from "react";
import { wsiFetch } from "@/lib/archive/client";

interface State {
  url: string | null;
  loading: boolean;
  error: Error | null;
}

/** Fetches an archive-relative path with auth attached and exposes it as a
 * Blob object URL suitable for `<img src>`. Plain `<img src="archive-url">`
 * can't work here: the archive requires an Authorization header, and only
 * `fetch()` can attach one - `<img>`/`<a>` tags have no such mechanism.
 */
export function useAuthedObjectUrl(path: string | null): State {
  const [state, setState] = useState<State>({
    url: null,
    loading: true,
    error: null,
  });

  // Resetting state for a new (or cleared) `path` prop - react.dev's own
  // data-fetching pattern (https://react.dev/learn/you-might-not-need-an-effect#fetching-data).
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!path) {
      setState({ url: null, loading: false, error: null });
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;
    setState({ url: null, loading: true, error: null });

    (async () => {
      try {
        const res = await wsiFetch(path);
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ url: objectUrl, loading: false, error: null });
      } catch (err) {
        if (!cancelled) {
          setState({
            url: null,
            loading: false,
            error: err instanceof Error ? err : new Error(String(err)),
          });
        }
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);
  /* eslint-enable react-hooks/set-state-in-effect */

  return state;
}
