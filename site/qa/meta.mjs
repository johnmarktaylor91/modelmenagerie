// usage: node qa/meta.mjs BASEURL OUT.json
// Share previews and structured data on the key pages: title, description, canonical address on the
// apex with a trailing slash, Open Graph and Twitter tags, the share image (fetched: PNG, 1200 by 630),
// and JSON-LD of the expected type. Fetches each page as the link-preview robots do too.
import fs from "node:fs";

const [base, out] = process.argv.slice(2);
const APEX = "https://modelmenagerie.ai";
const ROUTES = {
  home: { path: "/", jsonLd: "WebSite" },
  listing: { path: "/models/", jsonLd: null },
  model: { path: "/models/resnet18/", jsonLd: "CreativeWork" },
  dataset: { path: "/dataset/", jsonLd: "Dataset" },
  about: { path: "/about/", jsonLd: "AboutPage" },
  legal: { path: "/legal/", jsonLd: null },
  changelog: { path: "/changelog/", jsonLd: null },
};
const ROBOTS = {
  twitter: "Twitterbot/1.0",
  slack: "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
  imessage: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/601.2.4 (KHTML, like Gecko) Version/9.0.1 Safari/601.2.4 facebookexternalhit/1.1 Facebot Twitterbot/1.0",
};

function attr(tag, name) {
  return tag.match(new RegExp(`\\s${name}="([^"]*)"`))?.[1]?.replace(/&amp;/g, "&") ?? null;
}

function tags(html) {
  const meta = {};
  for (const [tag] of html.matchAll(/<meta\b[^>]*>/g)) {
    const key = attr(tag, "property") ?? attr(tag, "name");
    if (key) meta[key] = attr(tag, "content");
  }
  const canonical = [...html.matchAll(/<link\b[^>]*rel="canonical"[^>]*>/g)].map(([tag]) => attr(tag, "href"))[0] ?? null;
  const title = html.match(/<title>([^<]*)<\/title>/)?.[1] ?? null;
  const jsonLd = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map((m) => {
    try {
      return JSON.parse(m[1]);
    } catch {
      return { "@type": "INVALID" };
    }
  });
  return { meta, canonical, title, jsonLd };
}

/** Width and height of a PNG from its header, or null. */
function pngSize(bytes) {
  const signature = [0x89, 0x50, 0x4e, 0x47];
  if (bytes.length < 24 || !signature.every((b, i) => bytes[i] === b)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}

const results = {};
for (const [name, route] of Object.entries(ROUTES)) {
  const problems = [];
  const response = await fetch(base + route.path);
  const html = await response.text();
  const { meta, canonical, title, jsonLd } = tags(html);
  const expectCanonical = APEX + route.path;
  if (response.status !== 200) problems.push(`status ${response.status}`);
  if (!title) problems.push("no title");
  if (!meta.description || meta.description.length < 50) problems.push("description missing or under 50 characters");
  if (canonical !== expectCanonical) problems.push(`canonical ${canonical} is not ${expectCanonical}`);
  if (meta["og:url"] !== expectCanonical) problems.push(`og:url ${meta["og:url"]}`);
  for (const key of ["og:title", "og:description", "og:image", "og:image:alt", "twitter:card", "twitter:image"]) {
    if (!meta[key]) problems.push(`no ${key}`);
  }
  if (meta["twitter:card"] !== "summary_large_image") problems.push(`twitter:card ${meta["twitter:card"]}`);
  let image = null;
  if (meta["og:image"]) {
    if (!meta["og:image"].startsWith(`${APEX}/`) && !meta["og:image"].startsWith("https://")) problems.push("og:image is not an absolute https address");
    const imageUrl = meta["og:image"].startsWith(APEX) ? base + meta["og:image"].slice(APEX.length) : meta["og:image"];
    const imageResponse = await fetch(imageUrl);
    const bytes = new Uint8Array(await imageResponse.arrayBuffer());
    const size = pngSize(bytes);
    image = { url: meta["og:image"], status: imageResponse.status, type: imageResponse.headers.get("content-type"), bytes: bytes.length, ...size };
    if (imageResponse.status !== 200) problems.push(`share image status ${imageResponse.status}`);
    if (!size) problems.push("share image is not a PNG");
    else if (size.width !== 1200 || size.height !== 630) problems.push(`share image is ${size.width} by ${size.height}`);
    if (meta["og:image:width"] && Number(meta["og:image:width"]) !== size?.width) problems.push("og:image:width does not match the image");
  }
  const types = jsonLd.map((item) => item["@type"]);
  if (types.includes("INVALID")) problems.push("JSON-LD does not parse");
  if (route.jsonLd && !types.includes(route.jsonLd)) problems.push(`no ${route.jsonLd} JSON-LD (found ${types.join(", ") || "none"})`);
  results[name] = { path: route.path, title, description: meta.description, canonical, og_title: meta["og:title"], image, json_ld: types, problems };
}

// The robots that draw link previews get the same page and tags (no robot is turned away).
const robots = {};
for (const [name, agent] of Object.entries(ROBOTS)) {
  const response = await fetch(`${base}/`, { headers: { "User-Agent": agent } });
  const { meta } = tags(await response.text());
  robots[name] = { status: response.status, og_image: meta["og:image"] ?? null, ok: response.status === 200 && Boolean(meta["og:image"]) };
}

const problems = Object.values(results).reduce((sum, route) => sum + route.problems.length, 0) + Object.values(robots).filter((r) => !r.ok).length;
fs.writeFileSync(out, JSON.stringify({ problems, routes: results, robots }, null, 1));
console.log(JSON.stringify({ problems, routes: Object.fromEntries(Object.entries(results).map(([k, v]) => [k, v.problems])), robots }, null, 1));
