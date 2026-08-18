"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { getValidAccessToken, useAuth } from "@/lib/auth/client";

const markdownComponents: Components = {
  h1: (props) => (
    <h1
      className="mb-4 mt-8 text-2xl font-semibold text-zinc-900 first:mt-0 dark:text-zinc-100"
      {...props}
    />
  ),
  h2: (props) => (
    <h2
      className="mb-3 mt-8 text-lg font-semibold text-zinc-900 dark:text-zinc-100"
      {...props}
    />
  ),
  h3: (props) => (
    <h3
      className="mb-2 mt-6 text-base font-semibold text-zinc-900 dark:text-zinc-100"
      {...props}
    />
  ),
  p: (props) => (
    <p
      className="mb-4 leading-relaxed text-zinc-700 dark:text-zinc-300"
      {...props}
    />
  ),
  ul: (props) => (
    <ul
      className="mb-4 list-disc space-y-1 pl-6 text-zinc-700 dark:text-zinc-300"
      {...props}
    />
  ),
  ol: (props) => (
    <ol
      className="mb-4 list-decimal space-y-1 pl-6 text-zinc-700 dark:text-zinc-300"
      {...props}
    />
  ),
  a: (props) => (
    <a
      className="text-blue-600 underline hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    />
  ),
  code: ({ className, ...props }) => {
    const isBlock = /language-/.test(className ?? "");
    if (isBlock) {
      return <code className={className} {...props} />;
    }
    return (
      <code
        className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[0.85em] text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
        {...props}
      />
    );
  },
  pre: (props) => (
    <pre
      className="mb-4 overflow-x-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100 dark:bg-black"
      {...props}
    />
  ),
  blockquote: (props) => (
    <blockquote
      className="mb-4 border-l-2 border-zinc-300 pl-4 text-zinc-600 italic dark:border-zinc-700 dark:text-zinc-400"
      {...props}
    />
  ),
  hr: (props) => (
    <hr className="my-8 border-zinc-200 dark:border-zinc-800" {...props} />
  ),
  table: (props) => (
    <div className="mb-4 overflow-x-auto">
      <table className="w-full border-collapse text-sm" {...props} />
    </div>
  ),
  th: (props) => (
    <th
      className="border border-zinc-200 bg-zinc-50 px-3 py-2 text-left font-medium text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
      {...props}
    />
  ),
  td: (props) => (
    <td
      className="border border-zinc-200 px-3 py-2 text-zinc-700 dark:border-zinc-800 dark:text-zinc-300"
      {...props}
    />
  ),
};

export default function DocsPage() {
  const router = useRouter();
  const { loggedIn, ready } = useAuth();

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tokenCopyState, setTokenCopyState] = useState<
    "idle" | "copying" | "copied" | "error"
  >("idle");

  useEffect(() => {
    if (!ready) return;
    if (!loggedIn) {
      router.replace("/login");
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/API.md");
        if (!res.ok) throw new Error(`Failed to load docs (${res.status})`);
        const text = await res.text();
        if (!cancelled) setMarkdown(text);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : "Failed to load docs");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [ready, loggedIn, router]);

  async function handleCopyToken() {
    setTokenCopyState("copying");
    try {
      const token = await getValidAccessToken();
      await navigator.clipboard.writeText(token);
      setTokenCopyState("copied");
    } catch {
      setTokenCopyState("error");
    } finally {
      setTimeout(() => setTokenCopyState("idle"), 2000);
    }
  }

  if (!ready || (!markdown && !loadError)) {
    return (
      <div className="p-8 text-center text-zinc-500 dark:text-zinc-400">
        Loading...
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <Link
          href="/browse"
          className="text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
        >
          &larr; Back to browse
        </Link>
        <button
          onClick={handleCopyToken}
          className="rounded-full bg-zinc-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
        >
          {tokenCopyState === "copying" && "Copying..."}
          {tokenCopyState === "copied" && "Copied!"}
          {tokenCopyState === "error" && "Failed to copy"}
          {tokenCopyState === "idle" && "Copy access token"}
        </button>
      </div>

      {loadError ? (
        <p className="text-red-600">{loadError}</p>
      ) : (
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {markdown}
        </ReactMarkdown>
      )}
    </div>
  );
}
