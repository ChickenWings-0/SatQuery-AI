# SatQuery AI — Hosted Demo Runbook

How the public demo is served: the frontend on Vercel, the API and the GPU on the
dev box at home, reached over the internet through two tunnels. Everything here is
free-tier and needs no domain.

```
browser ──► https://<project>.vercel.app                  static frontend (Vercel)
   │
   ├─ start-up ──► https://<machine>.<tailnet>.ts.net/v1/health      Tailscale Funnel
   │               stable URL, the "signpost": reports fast_base_url
   │
   └─ everything else ──► https://<random>.trycloudflare.com/v1/…    Cloudflare quick tunnel
                          nearby edge, ~0.1 s per request; falls back to the Funnel URL
                                       │
                                       ▼
                    uvicorn on 127.0.0.1:8000 (dev box, RX 7900 XTX)
```

## Why two tunnels

| | Tailscale Funnel | Cloudflare quick tunnel |
|---|---|---|
| URL | stable (machine name in the tailnet) | random, **changes on every start** |
| Latency from India (measured 2026-11-25) | ~0.4 s warm, ~1.3 s new connection (ingress in Paris) | ~0.1 s warm, ~0.15 s new connection (Delhi edge) |
| 16 MB upload | 20.5 s | 1.6 s |
| Streamed responses (SSE) | streamed live | **buffered until the response ends** |
| Account / cost | free Tailscale account | none |

The Funnel URL is baked into the frontend build (`VITE_API_BASE`) because it never
changes. At start-up the frontend asks it for `/v1/health`, which reports the quick
tunnel's current URL as `fast_base_url` (read from `cloudflared`'s local metrics
server). The access gate health-checks that URL with the access code and, if it
answers within 4 s, sends all further traffic there. Any transport failure on the
quick tunnel moves the page back to the Funnel URL for the rest of the session.

Because the quick tunnel buffers streams, job progress is read there with the
short-poll form of the events endpoint (`GET /v1/jobs/{id}/events?poll=true`,
`API_CONTRACT.md` §4.4): each reply ends as soon as there is a new event, so
progress still arrives live. On the Funnel URL the ordinary SSE stream is used.

## Access code and GPU queue

The API is reachable by anyone who has its URL, so two server-side guards apply
(`API_CONTRACT.md` §4.12):

- **Access code.** With `SATQUERY_ACCESS_CODE` set, every `/v1` request must carry
  a matching code in the `X-SatQuery-Code` header (or `?code=` for URLs an `<img>`
  loads). The frontend shows a code screen first and remembers the code in the
  browser. Ten wrong codes from one IP lock it out for 15 minutes. Several codes can
  be issued, comma-separated, and revoked one at a time.
- **GPU queue.** One analysis runs on the GPU at a time
  (`SATQUERY_MAX_CONCURRENT_JOBS=1`); the rest wait in arrival order. Past four
  running or waiting (`SATQUERY_MAX_ACTIVE_JOBS=4`) a new request gets
  `429 GPU_BUSY` with a readable message.

The code lives only in the git-ignored `.env`. Changing it needs an API restart, not
a frontend redeploy.

## One-time setup

### 1. Backend `.env` additions

```bash
SATQUERY_ACCESS_CODE=<code>[,<another code>]
SATQUERY_FAST_PATH_METRICS_URL=http://127.0.0.1:20241/quicktunnel
```

### 2. Tailscale Funnel (stable URL)

```bash
sudo dnf config-manager addrepo --from-repofile=https://pkgs.tailscale.com/stable/fedora/tailscale.repo
sudo dnf install -y tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up                       # log in once in the browser
sudo tailscale funnel --bg 8000         # first run prints a link to enable Funnel
sudo tailscale set --accept-dns=false   # see "Troubleshooting": keeps the site usable on this box
```

`tailscale funnel status` shows the public URL. The Funnel configuration persists
across reboots.

### 3. Services (systemd user units, start at boot without a login)

`~/.config/systemd/user/satquery-api.service`:

```ini
[Unit]
Description=SatQuery AI API (public demo backend)
After=network-online.target

[Service]
WorkingDirectory=%h/SatQuery-AI
ExecStart=/bin/bash -c 'source scripts/rocm_env.sh && exec .venv/bin/uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 >> logs/demo_api.log 2>&1'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/satquery-tunnel.service`:

```ini
[Unit]
Description=Cloudflare quick tunnel to the SatQuery API (fast path; URL changes per start)
After=network-online.target satquery-api.service

[Service]
ExecStart=/usr/local/bin/cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 --url http://127.0.0.1:8000
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/satquery-awake.service` (keeps the box from sleeping):

```ini
[Unit]
Description=Keep the PC awake while the SatQuery demo is served

[Service]
ExecStart=/usr/bin/systemd-inhibit --what=sleep:idle --who=SatQuery --why="Serving the SatQuery demo" sleep infinity
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now satquery-api satquery-tunnel satquery-awake
loginctl enable-linger "$USER"          # start them at boot, before anyone logs in
```

The API unit runs `.venv/bin/uvicorn` directly — never `uv run` on this box
(`AI_HANDOFF/05`).

### 4. Vercel (frontend)

Import the GitHub repository, set **Root Directory** to `frontend`, and add one
environment variable:

| Variable | Value |
|---|---|
| `VITE_API_BASE` | the Funnel URL, no trailing slash |

`frontend/vercel.json` holds the rest: the Vite build with
`VITE_ONLINE_FEATURES_DEFAULT=true` (so the Maps imagery search works without a
trip to Settings) and a rewrite of every path to `index.html` so client-side routes
survive a refresh. Every push to `main` redeploys in about a minute.

## Day-to-day

| Task | Command |
|---|---|
| Status of everything | `systemctl --user status satquery-api satquery-tunnel satquery-awake` |
| Restart the API (e.g. after changing `.env`) | `systemctl --user restart satquery-api` |
| Current fast URL | `curl -s 127.0.0.1:20241/quicktunnel` |
| API log | `tail -f logs/demo_api.log` |
| Public check (with the code) | `curl -H "X-SatQuery-Code: <code>" https://<machine>.<tailnet>.ts.net/v1/health` |
| Stop serving | `systemctl --user stop satquery-tunnel satquery-api satquery-awake` and `sudo tailscale funnel --bg off 8000` |

**After a reboot nothing needs doing.** Tailscale and all three units come back on
their own; the quick tunnel gets a new URL and the frontend discovers it. The first
question after an API restart takes ~30 s while the model loads onto the GPU; later
ones take a few seconds. Ask one question before presenting.

## Pre-presentation checklist

1. Open the site on a phone **on mobile data** and answer one question end to end.
2. `curl -s 127.0.0.1:20241/quicktunnel` returns a hostname, and `/v1/health` through
   the Funnel URL reports the same one as `fast_base_url`.
3. The presenting laptop is not logged into the tailnet (or has run
   `tailscale set --accept-dns=false`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser console: *blocked by CORS policy: Permission was denied for this request to access the … address space* | That device is logged into the tailnet, so MagicDNS resolves the Funnel name to a private `100.x` address and Chrome's local-network protection blocks the public site from calling it. Run `sudo tailscale set --accept-dns=false` on that device, or disconnect Tailscale there. |
| Code screen never appears; the app shows "Could not reach the SatQuery API" | The API or Funnel is down: check `systemctl --user status satquery-api` and `tailscale funnel status`. |
| Everything works but feels slow (~1 s per click) | Traffic is on the Funnel URL: the quick tunnel is down or was not reported. Check `satquery-tunnel` and `SATQUERY_FAST_PATH_METRICS_URL`; reload the page once it is back. |
| `429 GPU_BUSY` | Four analyses are already running or waiting. Wait for them, or raise `SATQUERY_MAX_ACTIVE_JOBS`. |
| `429 ACCESS_LOCKED` | Ten wrong codes from that network in 15 minutes; wait it out or restart the API. |
| The first answer takes ~30 s | The model was loading after an API restart. |
