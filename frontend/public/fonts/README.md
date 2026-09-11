# Typefaces

Four `woff2` files, copied here rather than imported, so they have stable URLs
that `index.html` can `<link rel="preload">` — Vite hashes anything imported
through the module graph, and you cannot preload a filename you do not know
until build time.

They are **not** hand-collected binaries. They come from the pinned fontsource
packages in `devDependencies`, and `npm run fonts:sync` re-copies them:

| Family                | Package                            | Licence                    |
|-----------------------|------------------------------------|----------------------------|
| `Inter Variable`      | `@fontsource-variable/inter`       | `LICENSE-Inter.txt`        |
| `Geist Mono Variable` | `@fontsource-variable/geist-mono`  | `LICENSE-GeistMono.txt`    |

Both are SIL OFL 1.1, which requires the licence and copyright notice to travel
*with* the font files — which is why the two `LICENSE-*.txt` files sit here and
ship in the build rather than being tidied away into the repo root.

Both are variable fonts carrying the full 100–900 weight axis in one file, so
the three weights the UI uses (400 / 500 / 600) cost one download each, not
three. Only the Latin and Latin-Ext subsets are shipped; the `unicode-range`
declarations in `src/styles/theme.css` mean an English page never fetches
Latin-Ext, and the other subsets the packages contain (Cyrillic, Greek,
Vietnamese) fall back to the system stack. Add them here and declare their
ranges if that stops being acceptable.

To update: bump the package, run `npm run fonts:sync`, then re-derive the
fallback metrics in `theme.css` — a new release can change `unitsPerEm` or the
vertical metrics, which is exactly the reflow those overrides exist to prevent.
