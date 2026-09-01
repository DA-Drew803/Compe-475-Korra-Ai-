import { NextResponse } from "next/server";

export async function POST() {
  const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://backend:8000";
  const response = await fetch(`${backendUrl}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store"
  });
  const data = await response.json();
  return NextResponse.json(data, { status: response.status });
}
