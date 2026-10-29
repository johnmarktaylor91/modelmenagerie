import type { APIRoute } from "astro";
import { getModelIndexFiles } from "@/lib/modelIndex";

/** Emit the listing cards' picture keys, in index row order. */
export const GET: APIRoute = () =>
  new Response(JSON.stringify(getModelIndexFiles().thumbs), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
