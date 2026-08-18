"use client";

import { useState } from "react";
import { useAuthedObjectUrl } from "@/hooks/useAuthedObjectUrl";
import { downloadSlide } from "@/lib/archive/download";
import { config } from "@/lib/config";

function slideIdFromPath(path: string): string {
  return path.split("/").pop() ?? path;
}

/** xOpat is a separate, already-running viewer application hosted on the
 * same domain as the archive API (not something this app renders itself -
 * see README). It takes a `slides=` query param with the slide id. */
function xopatUrlFor(slideId: string): string | null {
  if (!config.wsiArchive.baseUrl) return null;
  const origin = new URL(config.wsiArchive.baseUrl).origin;
  return `${origin}/xopat/?slides=${encodeURIComponent(slideId)}`;
}

export function FileCard({ name, path }: { name: string; path: string }) {
  const slideId = slideIdFromPath(path);
  const xopatUrl = xopatUrlFor(slideId);
  const { url: thumbUrl } = useAuthedObjectUrl(
    `slides/thumbnail/max_size/480/480?slide_id=${encodeURIComponent(slideId)}`
  );
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  async function handleDownload() {
    setDownloading(true);
    setDownloadError(null);
    try {
      await downloadSlide(slideId);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="relative flex flex-col items-center gap-2 rounded-lg border border-zinc-200 p-3 text-center has-[.thumb-zone:hover]:z-20 dark:border-zinc-800">
      <div className="thumb-zone group/thumb flex h-24 w-24 items-center justify-center overflow-hidden rounded-md bg-zinc-100 dark:bg-zinc-800">
        {/* Object URL from an authenticated fetch, not a plain src="archive
            url" - the archive requires a Bearer header that only fetch()
            can attach. Same image scales up on hover (origin-bottom, grows
            away from the controls below it), no second request on hover. */}
        {thumbUrl && (
          <img
            src={thumbUrl}
            alt={name}
            className="h-24 w-24 origin-bottom rounded-md object-cover shadow-sm transition-transform duration-150 ease-out group-hover/thumb:scale-[3] group-hover/thumb:shadow-2xl"
          />
        )}
      </div>
      <span
        className="w-full truncate text-sm text-zinc-700 dark:text-zinc-300"
        title={name}
      >
        {name}
      </span>
      <div className="mt-1 flex gap-2">
        {xopatUrl && (
          <a
            href={xopatUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-zinc-300 px-3 py-1 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            Open in xOpat
          </a>
        )}
        <button
          onClick={handleDownload}
          disabled={downloading}
          className="rounded-full bg-zinc-900 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
        >
          {downloading ? "Downloading..." : "Download"}
        </button>
      </div>
      {downloadError && (
        <span className="text-xs text-red-600">{downloadError}</span>
      )}
    </div>
  );
}
