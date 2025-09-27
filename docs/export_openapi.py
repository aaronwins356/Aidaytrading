#!/usr/bin/env python3
"""Export the FastAPI OpenAPI specification to docs/openapi.json."""
from __future__ import annotations

import json
from pathlib import Path

from app.main import app
from fastapi.testclient import TestClient


def main() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    output_path = Path(__file__).resolve().parent / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OpenAPI document written to {output_path}")


if __name__ == "__main__":
    main()
