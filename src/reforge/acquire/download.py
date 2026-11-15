"""Reference acquisition: download a viral reel (by URL) to storage for deconstruction.

This closes the discovery loop: an outlier search (e.g. vidIQ) gives a reel URL, this fetches the
actual MP4 so we can deconstruct it (Instagram is auth-walled, so a downloader is required).

We use yt-dlp (free, scriptable) with the venv binary. Instagram may require session
cookies at volume; if `settings.ytdlp_cookies_from_browser` is set we pass
`--cookies-from-browser <value>` (empty = no cookies, which works for many public reels).

ToS note: downloading other creators' reels is against Instagram's ToS and is a gray area.
Reforge uses them as STRUCTURAL references only (cut timing, format) and renders with
Logan's OWN footage. Never republish someone else's footage or audio.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess

# instagram.com/{reel|reels|p|tv}/<shortcode>[/][?query]
_SHORTCODE_RE = re.compile(
    r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", re.IGNORECASE
)

_YTDLP = ".venv/bin/yt-dlp"


def shortcode_from_url(url: str | None) -> str | None:
    """Extract the reel/post shortcode from an Instagram URL, or None if not an IG url.

    Handles /reel/CODE, /reel/CODE/, /reels/CODE, /p/CODE, /tv/CODE, with or without a
    trailing slash or query string. Pure (no I/O).
    """
    if not url:
        return None
    m = _SHORTCODE_RE.search(url)
    return m.group(1) if m else None


def download_reel(url: str, dest_dir: str = "library/references") -> str:
    """Download an IG reel by URL via yt-dlp; return the path to the resulting .mp4.

    Names the file by shortcode (yt-dlp merges to <shortcode>.mp4). Raises RuntimeError with
    the yt-dlp stderr tail on failure or if no mp4 lands. Cookies are added when
    ``settings.ytdlp_cookies_from_browser`` is set (IG may need this at volume).
    """
    from ..config import settings

    shortcode = shortcode_from_url(url)
    if not shortcode:
        raise RuntimeError(f"not an Instagram reel/post URL: {url!r}")

    os.makedirs(dest_dir, exist_ok=True)
    out_tmpl = os.path.join(dest_dir, f"{shortcode}.%(ext)s")
    cmd = [_YTDLP, "-o", out_tmpl]
    cookies = getattr(settings, "ytdlp_cookies_from_browser", "")
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    cmd.append(url)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    mp4 = os.path.join(dest_dir, f"{shortcode}.mp4")
    if not os.path.exists(mp4):
        # yt-dlp may leave a non-mp4 container if the merge step didn't run - take any file.
        others = [p for p in glob.glob(os.path.join(dest_dir, f"{shortcode}.*"))]
        if others:
            mp4 = others[0]
        else:
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError(f"yt-dlp failed for {url} (rc={proc.returncode}):\n{tail}")
    return mp4
