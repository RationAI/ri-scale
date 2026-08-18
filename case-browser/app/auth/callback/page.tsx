"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { handleCallback } from "@/lib/auth/client";

function CallbackInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    handleCallback(searchParams)
      .then(() => router.replace("/browse"))
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Login failed")
      );
    // Intentionally run once - searchParams is stable for the lifetime of
    // this one-shot callback handling.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col items-center justify-center gap-4 px-6 text-center">
      {error ? (
        <>
          <p className="text-sm text-red-600">{error}</p>
          <a href="/login" className="text-sm underline">
            Back to login
          </a>
        </>
      ) : (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Signing you in...
        </p>
      )}
    </div>
  );
}

export default function CallbackPage() {
  return (
    <Suspense fallback={null}>
      <CallbackInner />
    </Suspense>
  );
}
