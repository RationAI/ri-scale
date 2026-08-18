import Link from "next/link";
import type { Breadcrumb } from "@/lib/archive/browse";

export function Breadcrumbs({ crumbs }: { crumbs: Breadcrumb[] }) {
  return (
    <nav className="flex flex-wrap items-center gap-1 text-sm text-zinc-500 dark:text-zinc-400">
      <Link
        href="/browse"
        className="hover:text-zinc-900 dark:hover:text-zinc-100"
      >
        root
      </Link>
      {crumbs.map((crumb, i) => (
        <span key={crumb.path} className="flex items-center gap-1">
          <span className="text-zinc-300 dark:text-zinc-600">/</span>
          {i === crumbs.length - 1 ? (
            <span className="font-medium text-zinc-900 dark:text-zinc-100">
              {crumb.name}
            </span>
          ) : (
            <Link
              href={`/browse/${crumb.path}`}
              className="hover:text-zinc-900 dark:hover:text-zinc-100"
            >
              {crumb.name}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}
