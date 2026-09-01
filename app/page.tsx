"use client";

import { useState } from "react";

export default function Home() {
  const [response, setResponse] = useState("");

  const callBackend = async () => {
    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/invoke`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: "Search the web for AI news"
      }),
    });

    const data = await res.json();
    setResponse(JSON.stringify(data));
  };

  return (
    <div>
      <h1>Korra UI</h1>
      <button onClick={callBackend}>Run Agent</button>
      <pre>{response}</pre>
    </div>
  );
}