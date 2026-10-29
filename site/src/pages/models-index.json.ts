import type { APIRoute } from "astro";
import { getModelIndexFiles } from "@/lib/modelIndex";

/** Emit the compact model index for the header search and the listing's search, filters and sorts. */
export const GET: APIRoute = () =>
  new Response(JSON.stringify(getModelIndexFiles().index), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
