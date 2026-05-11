import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent / "data" / "runtime"
GENERAL_CACHE = RUNTIME_DIR / "shenzhen_hospitals.json"
PSYCH_CACHE = RUNTIME_DIR / "shenzhen_psychiatric_hospitals.json"

GENERAL_SOURCE_URL = "https://wjw.sz.gov.cn/bmfw/wycx/fwyl/yycx/index.html"
PSYCH_SOURCE_URL = "https://wjw.sz.gov.cn/bmfw/wycx/fwyl/syjskdyy/index.html"


def _ensure_runtime_dir():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_html(url: str) -> str:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "-A",
            "Mozilla/5.0 (compatible; MindScapeClinicalOS/1.0)",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"curl failed for {url}")
    return result.stdout


def _read_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(cache_path: Path, payload: dict):
    _ensure_runtime_dir()
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _iter_json_records(html: str):
    matched = False
    for match in re.finditer(r'var\s+one_\d+\s*=\s*(\{.*?\});', html, re.DOTALL):
        matched = True
        chunk = match.group(1)
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            continue

    if matched:
        return

    for match in re.finditer(r'\{"jgmc":.*?\}', html, re.DOTALL):
        chunk = match.group(0)
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            continue


def _is_hospital_like(name: str) -> bool:
    return any(
        token in name
        for token in [
            "医院",
            "医疗中心",
            "妇幼保健院",
            "慢性病防治院",
            "护理院",
            "门诊部",
        ]
    )


def _normalize_general_record(record: dict) -> dict | None:
    name = (record.get("jgmc") or "").strip()
    if not name or not _is_hospital_like(name):
        return None
    return {
        "name": name,
        "level": (record.get("jb") or "").strip(),
        "address": (record.get("dz") or "").strip(),
        "phone": (record.get("dh") or "").strip(),
        "district": (record.get("xzqh") or "").strip(),
        "ownership": (record.get("jgflgldm") or "").strip(),
        "category_code": (record.get("wsjglbdm") or "").strip(),
    }


def _normalize_psych_record(record: dict) -> dict | None:
    name = (record.get("jgmc") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "address": (record.get("dz") or "").strip(),
        "district": (record.get("dq") or "").strip(),
    }


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for record in records:
        key = (record.get("name", ""), record.get("address", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def load_shenzhen_hospital_directory(limit: int = 120, refresh: bool = False) -> dict:
    cached = None if refresh else _read_cache(GENERAL_CACHE)
    if cached:
        hospitals = cached.get("hospitals", [])
        if len(hospitals) >= min(limit, 100):
            return {
                **cached,
                "hospitals": hospitals[:limit],
            }

    try:
        html = _fetch_html(GENERAL_SOURCE_URL)
        hospitals = _dedupe_records(
            [
                normalized
                for normalized in (
                    _normalize_general_record(record)
                    for record in _iter_json_records(html)
                )
                if normalized
            ]
        )
        payload = {
            "source_url": GENERAL_SOURCE_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total_count": len(hospitals),
            "hospitals": hospitals,
        }
        _write_cache(GENERAL_CACHE, payload)
        return {
            **payload,
            "hospitals": hospitals[:limit],
        }
    except Exception:
        if cached:
            hospitals = cached.get("hospitals", [])
            return {
                **cached,
                "hospitals": hospitals[:limit],
            }
        return {
            "source_url": GENERAL_SOURCE_URL,
            "fetched_at": "",
            "total_count": 0,
            "hospitals": [],
        }


def load_shenzhen_psychiatric_directory(refresh: bool = False) -> dict:
    cached = None if refresh else _read_cache(PSYCH_CACHE)
    if cached:
        return cached

    try:
        html = _fetch_html(PSYCH_SOURCE_URL)
        hospitals = _dedupe_records(
            [
                normalized
                for normalized in (
                    _normalize_psych_record(record)
                    for record in _iter_json_records(html)
                )
                if normalized
            ]
        )
        payload = {
            "source_url": PSYCH_SOURCE_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total_count": len(hospitals),
            "hospitals": hospitals,
        }
        _write_cache(PSYCH_CACHE, payload)
        return payload
    except Exception:
        if cached:
            return cached
        return {
            "source_url": PSYCH_SOURCE_URL,
            "fetched_at": "",
            "total_count": 0,
            "hospitals": [],
        }


def choose_emergency_hospital(
    district: str = "",
    preferred_hospital: str = "",
) -> dict:
    psych_directory = load_shenzhen_psychiatric_directory()
    hospitals = psych_directory.get("hospitals", [])

    if preferred_hospital:
        for hospital in hospitals:
            if hospital["name"] == preferred_hospital:
                return hospital

    if district:
        for hospital in hospitals:
            if hospital.get("district") == district:
                return hospital

    return hospitals[0] if hospitals else {
        "name": preferred_hospital or "Shenzhen psychiatric-capable hospital",
        "address": "",
        "district": district,
    }
