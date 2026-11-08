/**
 * A private contact address for the About and Legal pages. The site shows no personal address:
 * corrections and removal requests go to the public repository's issues (CONTACT in site.ts).
 * Set PUBLIC_CONTACT_EMAIL at build time only for a non-personal project address.
 */
export const CONTACT_EMAIL: string = import.meta.env.PUBLIC_CONTACT_EMAIL ?? "";
