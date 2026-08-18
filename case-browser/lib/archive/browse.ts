import { wsiFetch } from "./client";

export class PathAccessError extends Error {
  constructor(message = "Path not found or not accessible") {
    super(message);
    this.name = "PathAccessError";
  }
}

export interface Breadcrumb {
  name: string;
  path: string;
}

export function breadcrumbsFor(relativePath: string): Breadcrumb[] {
  if (!relativePath) return [];
  const parts = relativePath.split("/");
  let acc = "";
  return parts.map((name) => {
    acc = acc ? `${acc}/${name}` : name;
    return { name, path: acc };
  });
}

export interface FileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
}

interface RemoteCase {
  id: string;
  local_id: string;
  slides: string[];
}

function orgPrefixOf(id: string): string {
  return id.split(".")[0];
}

/** This app only ever exposes datasets under the `ratio` org - the archive
 * also has `mmci.mican`, which users of this app must not be able to reach
 * at all. Matched on just the first dot-segment (not the full `ratio.link`
 * site prefix) so any other `ratio.*` sub-dataset stays included too.
 * Filtered right here, at the one place all case data enters the app, so
 * it's a real access restriction (an unknown/rejected case id, same as any
 * other invalid path) rather than just something hidden in the UI.
 */
const ALLOWED_ORG = "ratio";

const CACHE_TTL_MS = 60_000;
let casesCache: { data: RemoteCase[]; fetchedAt: number } | null = null;

async function listCases(): Promise<RemoteCase[]> {
  if (casesCache && Date.now() - casesCache.fetchedAt < CACHE_TTL_MS) {
    return casesCache.data;
  }

  const res = await wsiFetch("cases/");
  if (!res.ok) {
    throw new Error(`Failed to list cases from WSI archive (${res.status})`);
  }
  const all = (await res.json()) as RemoteCase[];
  const data = all.filter((c) => orgPrefixOf(c.id) === ALLOWED_ORG);
  casesCache = { data, fetchedAt: Date.now() };
  return data;
}

export type RemoteLocation =
  | { kind: "root" }
  | { kind: "case"; caseId: string; slides: string[] }
  | { kind: "slide"; caseId: string; slideId: string };

export function relativePathOf(location: RemoteLocation): string {
  switch (location.kind) {
    case "root":
      return "";
    case "case":
      return location.caseId;
    case "slide":
      return `${location.caseId}/${location.slideId}`;
  }
}

/**
 * Resolves a case/slide path against the real (already site-filtered) case
 * list from the archive, rejecting anything that doesn't correspond to an
 * actual known case or slide - so a request can't smuggle an arbitrary
 * slide_id (including one from the excluded `mmci.mican` dataset) through
 * to the archive by way of a path that doesn't actually resolve.
 */
export async function resolveRemotePath(
  segments: string | string[] | undefined
): Promise<RemoteLocation> {
  const raw = Array.isArray(segments) ? segments : (segments ?? "").split("/");

  let parts: string[];
  try {
    parts = raw.map((s) => decodeURIComponent(s)).filter((s) => s.length > 0);
  } catch {
    throw new PathAccessError();
  }

  if (parts.length === 0) return { kind: "root" };
  if (parts.length > 2) throw new PathAccessError();

  const cases = await listCases();
  const caseId = parts[0];
  const matchedCase = cases.find((c) => c.id === caseId);
  if (!matchedCase) throw new PathAccessError();
  if (parts.length === 1) {
    return { kind: "case", caseId, slides: matchedCase.slides };
  }

  const slideId = parts[1];
  if (!matchedCase.slides.includes(slideId)) throw new PathAccessError();
  return { kind: "slide", caseId, slideId };
}

export async function listEntries(
  location: RemoteLocation
): Promise<FileEntry[]> {
  const cases = await listCases();

  switch (location.kind) {
    case "root": {
      return cases
        .map((c) => ({
          name: c.local_id || c.id,
          path: c.id,
          isDirectory: true,
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
    }
    case "case": {
      return location.slides.map((slideId) => ({
        name: slideId,
        path: `${location.caseId}/${slideId}`,
        isDirectory: false,
      }));
    }
    case "slide":
      throw new Error("A slide is a file, not a folder - nothing to list");
  }
}
