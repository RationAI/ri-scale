import { wsiFetch } from "./client";

/** Downloads a slide's zip (the archive's own multi-file storage, already
 * zipped server-side) and saves it via a temporary object-URL link.
 *
 * Known tradeoff: this buffers the whole file in browser memory before
 * saving (`await res.blob()`) - unlike a server-streamed download, there's
 * no stream-straight-to-disk here. Tested fine with a real 1.1GB slide;
 * the File System Access API could stream this properly in Chromium, but
 * isn't supported in Firefox/Safari, so it's not used here.
 */
export async function downloadSlide(slideId: string): Promise<void> {
  const res = await wsiFetch(
    `slides/download?slide_id=${encodeURIComponent(slideId)}`
  );
  if (!res.ok) {
    throw new Error(`Download failed (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slideId}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}
