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
  slides: string[];
}

function orgPrefixOf(id: string): string {
  return id.split(".")[0];
}

/** This app only exposes datasets under these orgs - the archive also has
 * `mmci.mican`, which users of this app must not be able to reach at all.
 * Matched on just the first dot-segment (not the full site prefix) so any
 * other `ratio.*` / `synth.*` sub-dataset stays included too.
 * Filtered right here, at the one place all case data enters the app, so
 * it's a real access restriction (an unknown/rejected case id, same as any
 * other invalid path) rather than just something hidden in the UI.
 */
const ALLOWED_ORGS = new Set(["ratio", "synth"]);

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
  const data = all.filter((c) => ALLOWED_ORGS.has(orgPrefixOf(c.id)));
  casesCache = { data, fetchedAt: Date.now() };
  return data;
}

/** A cohort is an `<org>.<site>` dataset, e.g. `ratio.link` or
 * `synth.link`. */
function cohortOf(caseId: string): string {
  return caseId.split(".").slice(0, 2).join(".");
}

export type RemoteLocation =
  | { kind: "root" }
  | { kind: "cohort"; cohort: string; slides: string[] };

export function relativePathOf(location: RemoteLocation): string {
  switch (location.kind) {
    case "root":
      return "";
    case "cohort":
      return location.cohort;
  }
}

/**
 * Resolves a cohort path against the real (already org-filtered) case list
 * from the archive, rejecting anything that isn't an actual known cohort.
 * Cases are an archive-side grouping with one slide each, so they are
 * flattened away: a cohort lists its slides directly.
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
  if (parts.length > 1) throw new PathAccessError();

  const cohort = parts[0];
  const cases = (await listCases()).filter((c) => cohortOf(c.id) === cohort);
  if (cases.length === 0) throw new PathAccessError();
  return {
    kind: "cohort",
    cohort,
    slides: cases.flatMap((c) => c.slides),
  };
}

export async function listEntries(
  location: RemoteLocation
): Promise<FileEntry[]> {
  switch (location.kind) {
    case "root": {
      const cases = await listCases();
      return [...new Set(cases.map((c) => cohortOf(c.id)))]
        .sort((a, b) => a.localeCompare(b))
        .map((cohort) => ({ name: cohort, path: cohort, isDirectory: true }));
    }
    case "cohort":
      return [...new Set(location.slides)]
        .sort((a, b) => a.localeCompare(b))
        .map((slideId) => ({
          name: slideId,
          path: `${location.cohort}/${slideId}`,
          isDirectory: false,
        }));
  }
}
