/**
 * Load an API image as pixels a canvas may read back.
 *
 * On the hosted deploy the API is another origin. An `<img>` request carries no
 * `Origin`, so the server answers it without CORS headers, and that answer is
 * cached as immutable — a later CORS `fetch` of the same URL is then served the
 * header-less copy and refused, and an `<img>` drawn onto a canvas taints it.
 * Either way the thumbnail, the SITREP scene and the GeoJSON mask vectoriser
 * got nothing. Fetching past the cache with CORS and decoding the bytes gives
 * an untainted bitmap on every origin, the same one included.
 */
export async function loadBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url, { cache: 'no-store', mode: 'cors' })
  if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`)
  return createImageBitmap(await response.blob())
}
