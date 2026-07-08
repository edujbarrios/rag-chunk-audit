"""Core audit helpers for rag-chunk-audit."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Any


PROMPT_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "reveal the system prompt",
    "show the system prompt",
    "developer message",
    "system message",
    "you are now",
    "act as",
    "jailbreak",
    "do not follow",
    "override instructions",
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^'\"\s]{8,}", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def audit_chunk(
    chunk: str | dict[str, object],
    *,
    min_chars: int = 40,
    max_chars: int = 4000,
    require_metadata: bool = False,
) -> list[dict[str, object]]:
    """Audit one chunk and return issue dictionaries."""

    normalized = _normalize_chunk(chunk, 0)
    return _audit_normalized_chunk(
        normalized,
        min_chars=min_chars,
        max_chars=max_chars,
        require_metadata=require_metadata,
        include_index=False,
    )


def audit_chunks(
    chunks: list[str | dict[str, object]],
    *,
    min_chars: int = 40,
    max_chars: int = 4000,
    require_metadata: bool = False,
) -> dict[str, object]:
    """Audit a list of chunks and return a compact report."""

    issues: list[dict[str, object]] = []
    normalized_chunks = [_normalize_chunk(chunk, index) for index, chunk in enumerate(chunks)]

    for normalized in normalized_chunks:
        issues.extend(
            _audit_normalized_chunk(
                normalized,
                min_chars=min_chars,
                max_chars=max_chars,
                require_metadata=require_metadata,
                include_index=True,
            )
        )

    issues.extend(_duplicate_issues(normalized_chunks))

    total_chunks = len(chunks)
    total_issues = len(issues)
    # Simple hygiene score: each issue costs half a chunk's share of the score.
    score = max(0, 100 - int((total_issues / max(total_chunks, 1)) * 50))

    return {
        "total_chunks": total_chunks,
        "total_issues": total_issues,
        "score": score,
        "issues": issues,
    }


def _normalize_chunk(chunk: Any, index: int) -> dict[str, Any]:
    if isinstance(chunk, str):
        return {
            "chunk_index": index,
            "text": chunk,
            "metadata": None,
            "valid": True,
            "is_dict": False,
        }

    if isinstance(chunk, dict):
        if "text" in chunk:
            text = chunk["text"]
        elif "content" in chunk:
            text = chunk["content"]
        else:
            return {
                "chunk_index": index,
                "text": "",
                "metadata": chunk.get("metadata"),
                "valid": False,
                "is_dict": True,
            }

        return {
            "chunk_index": index,
            "text": text if isinstance(text, str) else "",
            "metadata": chunk.get("metadata"),
            "valid": isinstance(text, str),
            "is_dict": True,
        }

    return {
        "chunk_index": index,
        "text": "",
        "metadata": None,
        "valid": False,
        "is_dict": False,
    }


def _audit_normalized_chunk(
    chunk: dict[str, Any],
    *,
    min_chars: int,
    max_chars: int,
    require_metadata: bool,
    include_index: bool,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    text = chunk["text"]

    if not chunk["valid"]:
        issues.append(
            _issue(
                "invalid_chunk",
                "high",
                "Chunk must be a string or a dictionary with text or content.",
                chunk["chunk_index"],
                include_index,
            )
        )
        return issues

    if require_metadata and chunk["is_dict"] and not chunk["metadata"]:
        issues.append(
            _issue(
                "missing_metadata",
                "medium",
                "Chunk is missing metadata.",
                chunk["chunk_index"],
                include_index,
            )
        )

    if not text.strip():
        issues.append(
            _issue(
                "empty_chunk",
                "medium",
                "Chunk is empty or whitespace only.",
                chunk["chunk_index"],
                include_index,
            )
        )
        return issues

    if len(text) < min_chars:
        issues.append(
            _issue(
                "too_short",
                "low",
                "Chunk is shorter than the minimum character length.",
                chunk["chunk_index"],
                include_index,
            )
        )

    if len(text) > max_chars:
        issues.append(
            _issue(
                "too_long",
                "medium",
                "Chunk is longer than the maximum character length.",
                chunk["chunk_index"],
                include_index,
            )
        )

    lowered = text.lower()
    if any(phrase in lowered for phrase in PROMPT_INJECTION_PHRASES):
        issues.append(
            _issue(
                "prompt_injection",
                "high",
                "Chunk contains instruction override language.",
                chunk["chunk_index"],
                include_index,
            )
        )

    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append(
            _issue(
                "secret_like_value",
                "high",
                "Chunk contains a secret-like value.",
                chunk["chunk_index"],
                include_index,
            )
        )

    if _has_repeated_boilerplate(text):
        issues.append(
            _issue(
                "repeated_boilerplate",
                "low",
                "Chunk contains repeated boilerplate-like text.",
                chunk["chunk_index"],
                include_index,
            )
        )

    if _has_low_information_density(text):
        issues.append(
            _issue(
                "low_information_density",
                "low",
                "Chunk has very low unique word density.",
                chunk["chunk_index"],
                include_index,
            )
        )

    return issues


def _duplicate_issues(chunks: list[dict[str, Any]]) -> list[dict[str, object]]:
    valid_chunks = [chunk for chunk in chunks if chunk["valid"] and chunk["text"].strip()]
    exact_seen: dict[str, int] = {}
    normalized_seen: dict[str, int] = {}
    issues: list[dict[str, object]] = []

    for chunk in valid_chunks:
        text = chunk["text"]
        index = chunk["chunk_index"]
        is_exact_duplicate = text in exact_seen

        if is_exact_duplicate:
            issues.append(
                _issue(
                    "exact_duplicate",
                    "medium",
                    "Chunk text exactly duplicates an earlier chunk.",
                    index,
                    True,
                )
            )
        else:
            exact_seen[text] = index

        normalized_text = _normalized_text(text)
        if normalized_text in normalized_seen and not is_exact_duplicate:
            issues.append(
                _issue(
                    "normalized_duplicate",
                    "medium",
                    "Chunk text duplicates an earlier chunk after normalization.",
                    index,
                    True,
                )
            )
        elif normalized_text:
            normalized_seen[normalized_text] = index

    return issues


def _normalized_text(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().strip().translate(table).split())


def _has_repeated_boilerplate(text: str) -> bool:
    words = _words(text)
    if len(words) >= 12:
        counts = Counter(words)
        if counts.most_common(1)[0][1] >= 6:
            return True

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    line_counts = Counter(lines)
    return any(count >= 3 for count in line_counts.values())


def _has_low_information_density(text: str) -> bool:
    words = _words(text)
    if len(words) < 12:
        return False

    return len(set(words)) / len(words) < 0.3


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _issue(
    issue_type: str,
    severity: str,
    message: str,
    chunk_index: int,
    include_index: bool,
) -> dict[str, object]:
    issue: dict[str, object] = {
        "type": issue_type,
        "severity": severity,
        "message": message,
    }
    if include_index:
        issue = {"chunk_index": chunk_index, **issue}
    return issue
