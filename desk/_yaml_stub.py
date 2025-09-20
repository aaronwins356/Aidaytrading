"""Minimal YAML loader/dumper used when PyYAML is unavailable."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value == "" or value == "null" or value == "~":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _preprocess(text: str) -> List[Tuple[int, str]]:
    processed: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        processed.append((indent, line.lstrip()))
    return processed


def _parse_block(lines: List[Tuple[int, str]], index: int, indent: int):
    if index >= len(lines):
        return None, index
    current_indent, content = lines[index]
    if content.startswith("- ") and current_indent == indent:
        items: List[Any] = []
        while index < len(lines):
            item_indent, item_content = lines[index]
            if item_indent != indent or not item_content.startswith("- "):
                break
            body = item_content[2:].strip()
            index += 1
            if body and ":" in body:
                key, value_part = body.split(":", 1)
                value_part = value_part.strip()
                item: Any
                if value_part:
                    item = {key.strip(): _coerce_scalar(value_part)}
                else:
                    nested, index = _parse_block(lines, index, indent + 2)
                    item = {key.strip(): nested}
            elif body:
                item = _coerce_scalar(body)
            else:
                nested, index = _parse_block(lines, index, indent + 2)
                item = nested
            # Merge additional nested mappings for dict items.
            if isinstance(item, dict):
                while index < len(lines) and lines[index][0] >= indent + 2:
                    nested, index = _parse_block(lines, index, indent + 2)
                    if isinstance(nested, dict):
                        item.update(nested)
                    elif nested is not None:
                        item.setdefault("items", nested)
            items.append(item)
        return items, index

    mapping: Dict[str, Any] = {}
    while index < len(lines):
        line_indent, line_content = lines[index]
        if line_indent < indent:
            break
        if line_content.startswith("- ") and line_indent == indent:
            # Start of a sequence inside a mapping value
            nested, index = _parse_block(lines, index, indent)
            return nested, index
        key, _, value_part = line_content.partition(":")
        key = key.strip()
        value_part = value_part.strip()
        index += 1
        if value_part:
            mapping[key] = _coerce_scalar(value_part)
        else:
            nested, index = _parse_block(lines, index, indent + 2)
            mapping[key] = nested
    return mapping, index


def safe_load(stream) -> Any:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)
    if not text:
        return {}
    lines = _preprocess(text)
    if not lines:
        return {}
    data, _ = _parse_block(lines, 0, lines[0][0])
    return data or {}


def safe_dump(data: Any, stream=None, sort_keys: bool = False) -> str:
    text = json.dumps(data, indent=2, sort_keys=sort_keys)
    if stream is not None:
        stream.write(text)
    return text


# expose in module-like namespace
class _Shim:
    safe_load = staticmethod(safe_load)
    safe_dump = staticmethod(safe_dump)


yaml = _Shim()
