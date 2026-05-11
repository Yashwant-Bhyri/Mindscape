"""
Optional press-RSS / NewsAPI headline digest — NOT the default source of truth for the MH forum.

Forum threads use curated seed + clinician posts unless MEDICAL_NEWS_FORUM_THREADS=true.
Weekly brief uses this only if MEDICAL_NEWS_BRIEF=true (off by default).
For literature-grounded lines, use WEEKLY_BRIEF_PUBMED (PubMed E-utilities) in the facade.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_CACHE_BRIEF: tuple[float, list[str]] | None = None
TTL_SECONDS = int(os.getenv("MEDICAL_NEWS_CACHE_SECONDS", "900"))
ENABLED = os.getenv("MEDICAL_NEWS_ENABLED", "false").lower() in ("1", "true", "yes")

# Curated, demo-safe feeds (public RSS). Failures are skipped individually.
RSS_FEEDS: list[tuple[str, str]] = [
    ("NIH News", "https://www.nih.gov/news-events/news-releases/rss.xml"),
    ("WHO", "https://www.who.int/rss-feeds/news-english.xml"),
    ("STAT", "https://www.statnews.com/feed/"),
    ("FDA Press announcements", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"),
    ("ScienceDaily Health", "https://www.sciencedaily.com/rss/health_medicine.xml"),
]

USER_AGENT = (
    "MindScapeClinicalDemo/1.0 (+https://github.com/) "
    "httpx; medical-news-ingest for clinician research surface"
)


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _iso_from_pub(pub: str | None) -> str:
    if not pub:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    pub = pub.strip()
    try:
        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        return dt.isoformat(timespec="seconds")
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(pub)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError):
        pass
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_rss_items(xml_bytes: bytes, source_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.debug("RSS parse error (%s): %s", source_label, exc)
        return out

    # RSS 2.0: channel/item; Atom: feed/entry
    for elem in root.iter():
        tag = _local_tag(elem.tag)
        if tag == "item":
            title_el = None
            link_el = None
            desc_el = None
            pub_el = None
            for child in list(elem):
                ct = _local_tag(child.tag)
                if ct == "title":
                    title_el = child
                elif ct == "link":
                    link_el = child
                elif ct == "description":
                    desc_el = child
                elif ct == "encoded" or str(child.tag).lower().endswith("encoded"):
                    desc_el = child
                elif ct in ("pubDate", "dc:date", "date"):
                    pub_el = child
            title = _clean_html(_text(title_el))
            link = (_text(link_el) or "").strip()
            body = _clean_html(_text(desc_el))[:1200]
            pub_raw = _text(pub_el)
            if title and link:
                out.append(_thread_from_article(source_label, title, link, body, pub_raw))
        elif tag == "entry":
            title_el = link_el = summary_el = pub_el = None
            for child in list(elem):
                ct = _local_tag(child.tag)
                if ct == "title":
                    title_el = child
                elif ct == "link":
                    link_el = child
                elif ct in ("summary", "content"):
                    summary_el = child
                elif ct in ("published", "updated"):
                    pub_el = child
            title = _clean_html(_text(title_el))
            href = ""
            if link_el is not None:
                href = (link_el.get("href") or _text(link_el) or "").strip()
            body = _clean_html(_text(summary_el))[:1200]
            pub_raw = _text(pub_el)
            if title and href:
                out.append(_thread_from_article(source_label, title, href, body, pub_raw))
    return out


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _thread_from_article(
    source_label: str,
    title: str,
    link: str,
    body: str,
    pub_raw: str,
) -> dict[str, Any]:
    url_hash = hashlib.sha256(link.encode()).hexdigest()[:14]
    created = _iso_from_pub(pub_raw)
    kind = "trend"
    flair = "Trend watch"
    tl = title.lower()
    if any(k in tl for k in ("trial", "study", "research", "published", "journal", "randomized")):
        kind = "research"
        flair = "Research share"
    return {
        "id": f"live-news-{url_hash}",
        "title": title[:240],
        "flair": flair,
        "kind": kind,
        "community": f"Live headlines · {source_label}",
        "author": source_label,
        "created": created,
        "date_label": created[:10],
        "body": body or f"Open the source link for full reporting — {source_label}.",
        "link": link,
        "link_title": "Read source article",
        "replies": [],
        "source": "medical_news",
        "live_headline": True,
    }


def _fetch_url(url: str, timeout: float = 12.0) -> bytes | None:
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code != 200:
                logger.debug("Feed HTTP %s %s", response.status_code, url)
                return None
            return response.content
    except Exception as exc:
        logger.debug("Feed fetch failed %s: %s", url, exc)
        return None


def _fetch_newsapi_health(limit: int) -> list[dict[str, Any]]:
    key = (os.getenv("NEWS_API_KEY") or "").strip()
    if not key:
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "health", "pageSize": min(limit, 30), "apiKey": key}
    try:
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url, params=params)
            if response.status_code != 200:
                logger.debug("NewsAPI status %s", response.status_code)
                return []
            data = response.json()
    except Exception as exc:
        logger.debug("NewsAPI error: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for art in data.get("articles") or []:
        title = (art.get("title") or "").strip()
        link = (art.get("url") or "").strip()
        body = _clean_html(art.get("description") or art.get("content") or "")[:1200]
        pub = art.get("publishedAt")
        if title and link:
            out.append(_thread_from_article("NewsAPI · Health", title, link, body, pub))
    return out


def _gather_rss(limit: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    per_feed = max(4, min(8, limit // max(1, len(RSS_FEEDS))))
    for label, feed_url in RSS_FEEDS:
        raw = _fetch_url(feed_url)
        if not raw:
            continue
        collected.extend(_parse_rss_items(raw, label)[:per_feed])
        if len(collected) >= limit:
            break
    return collected[:limit]


def _dedupe_by_link(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for t in threads:
        link = (t.get("link") or "").strip().lower()
        key = link or t.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def get_live_medical_headlines_as_threads(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Return forum-shaped thread dicts from live RSS (+ optional NewsAPI).
    Safe to call frequently; results are cached ~15 minutes.
    """
    global _CACHE
    if not ENABLED:
        return []

    limit = limit if limit is not None else int(os.getenv("MEDICAL_NEWS_MAX", "18"))
    now = time.time()
    if _CACHE and now - _CACHE[0] < TTL_SECONDS:
        return _CACHE[1][:limit]

    merged: list[dict[str, Any]] = []
    try:
        merged.extend(_fetch_newsapi_health(limit))
        merged.extend(_gather_rss(max(limit * 2, limit + 10)))
    except Exception as exc:
        logger.warning("Medical news aggregation failed: %s", exc)

    merged = _dedupe_by_link(merged)
    merged.sort(key=lambda x: x.get("created") or "", reverse=True)
    merged = merged[:limit]

    for row in merged:
        replies = row.get("replies") or []
        row["reply_count"] = len(replies)
        row["activity"] = "Live headline"

    _CACHE = (now, merged)
    return merged


def get_news_brief_lines(max_lines: int = 4) -> list[str]:
    """Short bullets for the weekly agent brief card (cached with headlines)."""
    global _CACHE_BRIEF
    now = time.time()
    if _CACHE_BRIEF and now - _CACHE_BRIEF[0] < TTL_SECONDS:
        return _CACHE_BRIEF[1][:max_lines]

    threads = get_live_medical_headlines_as_threads(limit=max_lines + 6)
    lines: list[str] = []
    for t in threads:
        src = (t.get("author") or "Headline").split("·")[-1].strip()
        title = (t.get("title") or "").strip()
        if title:
            lines.append(f"[{src}] {title}")
        if len(lines) >= max_lines:
            break

    _CACHE_BRIEF = (now, lines)
    return lines
