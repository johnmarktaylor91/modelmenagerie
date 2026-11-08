import type { APIRoute } from "astro";
import { getRosterIndexFile } from "@/lib/roster";

/** Emit the compact list of catalogued models with no diagram yet; loaded only on request. */
export const GET: APIRoute = () =>
  new Response(JSON.stringify(getRosterIndexFile()), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
