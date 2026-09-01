"use client";

import { FormEvent, useState } from "react";

type ChatMessage = {
  role: "human" | "ai";
  content: string;
};

export default function HomePage() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();

    setMessages((current) => [
      ...current,
      { role: "human", content: userText }
    ]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/runs/wait`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          assistant_id: "agent",
          input: {
            messages: [
              {
                role: "user",
                content: userText
              }
            ]
          }
        })
      });

      const data = await res.json();
      console.log("Backend response:", data);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${JSON.stringify(data)}`);
      }

      const outputMessages =
        data?.output?.messages ||
        data?.messages ||
        data?.values?.messages ||
        [];

      const lastMessage = outputMessages[outputMessages.length - 1];

      let aiContent = "No response returned.";

      if (typeof lastMessage?.content === "string") {
        aiContent = lastMessage.content;
      } else if (Array.isArray(lastMessage?.content)) {
        aiContent =
          lastMessage.content
            .map((item: any) =>
              typeof item === "string" ? item : item?.text || ""
            )
            .join(" ")
            .trim() || "No response returned.";
      }

      setMessages((current) => [
        ...current,
        { role: "ai", content: aiContent }
      ]);
    } catch (error: any) {
      console.error("Chat request failed:", error);
      setMessages((current) => [
        ...current,
        {
          role: "ai",
          content: `Request failed: ${error?.message || "Unknown error"}`
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1>Korra</h1>
      <p>Module 10 Section 3 containerized LangGraph + Next.js stack</p>

      <p>
        Try: Search the web for the latest AI news or Save this note to the
        database: buy lab parts
      </p>

      <div>
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`}>
            <strong>{message.role === "human" ? "You" : "Korra"}: </strong>
            {message.content}
          </div>
        ))}
      </div>

      <form onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Send a message to Korra"
        />
        <button type="submit" disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </main>
  );
}