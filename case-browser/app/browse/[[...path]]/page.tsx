"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { FolderCard } from "@/components/FolderCard";
import { FileCard } from "@/components/FileCard";
import { useAuth } from "@/lib/auth/client";
import {
  FileEntry,
  PathAccessError,
  RemoteLocation,
  breadcrumbsFor,
  listEntries,
  relativePathOf,
  resolveRemotePath,
} from "@/lib/archive/browse";

export default function BrowsePage() {
  const params = useParams<{ path?: string[] }>();
  const router = useRouter();
  const { loggedIn, ready } = useAuth();

  const [location, setLocation] = useState<RemoteLocation | null>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const pathKey = params.path?.join("/") ?? "";

  useEffect(() => {
    if (!ready) return;
    if (!loggedIn) {
      router.replace("/login");
      return;
    }

    // Resetting load state for a new path - react.dev's own data-fetching
    // pattern (https://react.dev/learn/you-might-not-need-an-effect#fetching-data).
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect */
    setLoading(true);
    setError(null);
    setNotFound(false);
    /* eslint-enable react-hooks/set-state-in-effect */

    (async () => {
      try {
        const loc = await resolveRemotePath(params.path);
        if (loc.kind === "slide") {
          // A slide is a file, not a browsable page - nothing to render here.
          if (!cancelled) setNotFound(true);
          return;
        }
        const ents = await listEntries(loc);
        if (!cancelled) {
          setLocation(loc);
          setEntries(ents);
        }
      } catch (err) {
        if (cancelled) return;
        if (err instanceof PathAccessError) {
          setNotFound(true);
        } else {
          setError(err instanceof Error ? err.message : "Failed to load");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, loggedIn, pathKey]);

  if (!ready || loading) {
    return (
      <div className="p-8 text-center text-zinc-500 dark:text-zinc-400">
        Loading...
      </div>
    );
  }

  if (notFound) {
    return (
      <div className="p-8 text-center text-zinc-500 dark:text-zinc-400">
        Not found.
      </div>
    );
  }

  if (error || !location) {
    return (
      <div className="p-8 text-center text-red-600">
        {error ?? "Something went wrong"}
      </div>
    );
  }

  const relativePath = relativePathOf(location);
  const crumbs = breadcrumbsFor(relativePath);
  const folders = entries.filter((e) => e.isDirectory);
  const files = entries.filter((e) => !e.isDirectory);

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
          WSI Case Viewer
        </h1>
        <div className="mt-2">
          <Breadcrumbs crumbs={crumbs} />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
          <Link
            href="/docs"
            className="rounded-full border border-zinc-300 px-3 py-1 font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            Script docs &amp; access token
          </Link>
        </div>
      </header>

      {entries.length === 0 ? (
        <p className="mt-16 text-center text-zinc-500 dark:text-zinc-400">
          This folder is empty.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
          {folders.map((folder) => (
            <FolderCard key={folder.path} name={folder.name} path={folder.path} />
          ))}
          {files.map((file) => (
            <FileCard key={file.path} name={file.name} path={file.path} />
          ))}
        </div>
      )}
    </div>
  );
}
