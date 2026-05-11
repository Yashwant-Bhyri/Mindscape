"""
Peer-reviewed literature pointers for the weekly agent brief (optional).

Uses NCBI E-utilities (no API key; polite-use tool + email via env).
Mental-health–focused query; returns recent PubMed titles — not Twitter-style news.

Enable with WEEKLY_BRIEF_PUBMED=true. Set NCBI_EMAIL for responsible use.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CACHE: tuple[float, list[str]] | None = None
TTL_SECONDS = int(os.getenv("PUBMED_BRIEF_CACHE_SECONDS", "900"))

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Peer-reviewed mental-health synthesis + trials (PubMed index). Kept MeSH-light for resilient E-search.
DEFAULT_QUERY = (
    '(psychiatry[tiab] OR "mental health"[tiab] OR depression[tiab] OR anxiety[tiab] '
    'OR "mood disorder"[tiab] OR psychosis[tiab]) '
    "AND (Review[Publication Type] OR Clinical Trial[Publication Type] OR Meta-Analysis[Publication Type])"
)

FALLBACK_QUERY = (
    "(psychiatry[tiab] OR depression[tiab] OR anxiety[tiab]) "
    "AND (Review[ptyp] OR Clinical Trial[ptyp] OR randomized controlled trial[tiab])"
)


def _esearch_ids(client: httpx.Client, params_base: dict[str, Any], term: str) -> list[str]:
    p = dict(params_base)
    p["term"] = term
    r = client.get(ESEARCH, params=p)
    if r.status_code != 200:
        logger.debug("PubMed esearch HTTP %s", r.status_code)
        return []
    data = r.json()
    er = data.get("esearchresult") or {}
    if er.get("ERROR"):
        logger.warning("PubMed esearch error: %s", er.get("ERROR"))
        return []
    return list(er.get("idlist") or [])


def _fetch_pubmed_brief(max_lines: int) -> list[str]:
    email = (os.getenv("NCBI_EMAIL") or "mindscape-forum-dev@localhost").strip()
    tool = "MindScapeClinicalForum"
    primary_q = (os.getenv("PUBMED_BRIEF_QUERY") or DEFAULT_QUERY).strip()

    params_base: dict[str, Any] = {
        "db": "pubmed",
        "retmax": min(max_lines + 6, 50),
        "sort": "pub+date",
        "retmode": "json",
        "tool": tool,
        "email": email,
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            id_list = _esearch_ids(client, params_base, primary_q)
            if not id_list:
                id_list = _esearch_ids(client, params_base, FALLBACK_QUERY)
            if not id_list:
                return []
            ids = ",".join(id_list[: max_lines + 4])
            r2 = client.get(
                ESUMMARY,
                params={
                    "db": "pubmed",
                    "id": ids,
                    "retmode": "json",
                    "tool": tool,
                    "email": email,
                },
            )
            if r2.status_code != 200:
                return []
            summary = r2.json().get("result") or {}

            order = summary.get("uids") or id_list
            lines: list[str] = []
            for uid in order:
                if uid in (None, "uids") or len(lines) >= max_lines:
                    continue
                article = summary.get(str(uid))
                if not isinstance(article, dict):
                    continue
                title = (article.get("title") or "").strip()
                title = re.sub(r"<[^>]+>", "", title).strip()
                if not title:
                    continue
                journal = (article.get("source") or "").strip()
                pubdate = (article.get("pubdate") or article.get("epubdate") or "").strip()
                tail = " · ".join(x for x in (journal, pubdate) if x)
                if tail:
                    lines.append(f"{title} ({tail})")
                else:
                    lines.append(title)

            return lines[:max_lines]
    except Exception as exc:
        logger.debug("PubMed brief failed: %s", exc)
        return []


def get_pubmed_psychiatry_brief_lines(max_lines: int = 4) -> list[str]:
    """Cached recent PubMed titles for the weekly brief card."""
    global _CACHE
    if os.getenv("WEEKLY_BRIEF_PUBMED", "").lower() not in ("1", "true", "yes"):
        return []

    now = time.time()
    if _CACHE and now - _CACHE[0] < TTL_SECONDS:
        return _CACHE[1][:max_lines]

    lines = _fetch_pubmed_brief(max_lines)
    _CACHE = (now, lines)
    return lines[:max_lines]
