"use client";

import { useState } from "react";
import { startLogin } from "@/lib/auth/client";

export default function LoginPage() {
  const [error, setError] = useState<string | null>(null);

  async function handleClick() {
    try {
      await startLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start login");
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col items-center justify-center gap-6 px-6 text-center">
      <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">
        WSI Case Viewer
      </h1>
      <p className="text-zinc-500 dark:text-zinc-400">
        Sign in with your Life Science AAI account to browse and download
        slide data from the BBMRI-ERIC WSI archive.
      </p>
      <button
        onClick={handleClick}
        className="rounded-full bg-zinc-900 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
      >
        Sign in with Life Science AAI
      </button>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
