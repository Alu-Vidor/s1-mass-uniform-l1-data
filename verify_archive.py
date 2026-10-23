"""Check the published snapshot without installing scientific dependencies."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "CHECKSUMS.json").read_text(encoding="utf-8"))
    failures = []
    total = 0
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(item["path"] + ": missing or invalid path")
            continue
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            failures.append(item["path"] + ": size or SHA-256 mismatch")
        total += len(data)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {len(manifest['files'])} files ({total:,} bytes).")


if __name__ == "__main__":
    main()
