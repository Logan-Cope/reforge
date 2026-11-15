"""App factory for the Reforge review/curation surface (FastAPI + Jinja2 + HTMX).

``create_app(queries=None)`` takes an injectable queries seam so route tests run
against a fake with no DB. Two env flags make the public demo (see ``create_app``):
``DEMO_FIXTURES`` and ``DEMO_READ_ONLY``. An optional invite gate (env ``INVITE_CODE``)
covers a private deploy: pass ``?code=`` once, get a cookie. ``/healthz`` stays open;
it doubles as the database keep-alive ping.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reforge.app.queries import Queries, proxy_url, reel_video_url, thumb_url

_INVITE_COOKIE = "rf_invite"


class RestoreRewrittenPath:
    """Put the visitor's original path back before anything routes on it.

    ``vercel.json`` rewrites every request to the single ``/api/index`` function,
    and that rewrite REPLACES the path - the function is invoked with
    ``/api/index`` whatever the visitor asked for, so every route 404s and
    ``/healthz`` stops matching its own invite-gate bypass. No incoming header
    carries the original path on this runtime, so the rewrite smuggles it through
    as ``?__p=<path>`` and this middleware restores it. Vercel appends the
    visitor's own query params alongside ``__p``, so ``?code=`` survives.

    Deployments built before ~2026-07-22 received the original path directly;
    when ``__p`` is absent the scope passes through untouched, so this is safe
    both locally (``scripts/serve_app.py``) and on older runtime behaviour.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            pairs = parse_qsl(scope.get("query_string", b"").decode("latin-1"),
                              keep_blank_values=True)
            original = next((v for k, v in pairs if k == "__p"), None)
            if original:
                if not original.startswith("/"):
                    original = "/" + original
                rest = urlencode([(k, v) for k, v in pairs if k != "__p"])
                scope = dict(scope,
                             path=original,
                             raw_path=original.encode("utf-8"),
                             query_string=rest.encode("latin-1"))
        await self.app(scope, receive, send)


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def env_flag(name: str) -> bool:
    """True for 1/true/yes/on (any case); unset or anything else is False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def create_app(queries=None) -> FastAPI:
    """Build the app. ``queries`` defaults to the live DB seam; tests inject a fake.

    ``DEMO_FIXTURES=1`` serves the bundled synthetic data (``reforge/demo``) with no
    database. ``DEMO_READ_ONLY=1`` refuses every non-GET request with 403 before any
    route runs, so a public demo can browse but never grade, schedule, run or post.
    """
    from reforge.config import settings

    read_only = env_flag("DEMO_READ_ONLY")
    fixtures = env_flag("DEMO_FIXTURES")

    app = FastAPI(title="Reforge")
    if queries is None:
        if fixtures:
            from reforge.app.fixtures import DEMO_DIR, FixtureQueries
            queries = FixtureQueries()
        else:
            queries = Queries()
    app.state.queries = queries
    app.state.read_only = read_only

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["thumb_url"] = thumb_url
    templates.env.filters["reel_video_url"] = reel_video_url
    templates.env.filters["proxy_url"] = proxy_url
    templates.env.globals["read_only"] = read_only
    app.state.templates = templates

    if fixtures:
        media = DEMO_DIR / "media"
        thumbs_dir, output_dir = str(media / "thumbs"), str(media / "output")
        proxies_dir = str(media / "proxies")
    else:
        thumbs_dir = os.path.join(settings.local_data_dir, "thumbs")
        output_dir = settings.output_dir or "output"
        proxies_dir = os.path.join(settings.local_data_dir, "proxies")
    app.mount("/thumbs", StaticFiles(directory=thumbs_dir, check_dir=False), name="thumbs")
    # local media for the watch-first review; absent on the cloud deploy (check_dir=False
    # keeps boot safe there - requests 404 and the templates degrade to stills/no player)
    app.mount("/media/output", StaticFiles(directory=output_dir, check_dir=False),
              name="media_output")
    app.mount("/media/proxies", StaticFiles(directory=proxies_dir, check_dir=False),
              name="media_proxies")
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
              name="static")

    @app.middleware("http")
    async def read_only_gate(request: Request, call_next):
        if read_only and request.method not in _SAFE_METHODS:
            return PlainTextResponse("This demo is read-only.", status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def invite_gate(request: Request, call_next):
        code = os.environ.get("INVITE_CODE", "")
        if not code or request.url.path == "/healthz":
            return await call_next(request)
        if request.cookies.get(_INVITE_COOKIE) == code:
            return await call_next(request)
        if request.query_params.get("code") == code:
            response = await call_next(request)
            response.set_cookie(_INVITE_COOKIE, code, max_age=30 * 24 * 3600, httponly=True)
            return response
        return PlainTextResponse("Invite code required - add ?code=... to the URL.",
                                 status_code=403)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "db": app.state.queries.ping()}

    from reforge.app.routes import router
    from reforge.app.schedule_routes import router as schedule_router
    app.include_router(router)
    app.include_router(schedule_router)
    # added last => outermost, so the invite gate and the router both see the
    # restored path rather than "/api/index"
    app.add_middleware(RestoreRewrittenPath)
    return app


app = create_app()  # uvicorn target: reforge.app.main:app
