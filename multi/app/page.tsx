// ./multi/app/page.tsx
"use client";
import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) {
        throw new Error("Failed to fetch response");
      }
      const data = await res.json();
      setResponse(data.answer);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-4">AI Query Interface</h1>
      <input
        type="text"
        className="border p-2 rounded w-96 text-black"
        placeholder="Ask something about deployments, external reports, or sentiment analysis..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded mt-4"
        onClick={handleQuery}
        disabled={loading}
      >
        {loading ? "Querying..." : "Ask AI"}
      </button>
      {response && (
        <div className="mt-4 p-4 border rounded bg-gray-100 w-96 text-black">
          <strong>Response:</strong>
          <p>{response}</p>
        </div>
      )}
      {error && <p className="text-red-500 mt-2">Error: {error}</p>}
    </div>
  );
}