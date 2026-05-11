"""
Keep CI deterministic: medical headline ingestion is optional and network-backed.
Demo deployments leave MEDICAL_NEWS_ENABLED at default (on).
"""

import os

os.environ.setdefault("MEDICAL_NEWS_ENABLED", "false")
os.environ.setdefault("MEDICAL_NEWS_FORUM_THREADS", "false")
os.environ.setdefault("MEDICAL_NEWS_BRIEF", "false")
os.environ.setdefault("WEEKLY_BRIEF_PUBMED", "false")
