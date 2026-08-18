import Link from "next/link";

export function FolderCard({ name, path }: { name: string; path: string }) {
  return (
    <Link
      href={`/browse/${path}`}
      className="flex flex-col items-center gap-2 rounded-lg border border-transparent p-3 text-center hover:border-zinc-200 hover:bg-zinc-50 dark:hover:border-zinc-800 dark:hover:bg-zinc-900"
    >
      <div className="flex h-24 w-24 items-center justify-center text-5xl">
        📁
      </div>
      <span
        className="w-full truncate text-sm text-zinc-700 dark:text-zinc-300"
        title={name}
      >
        {name}
      </span>
    </Link>
  );
}
