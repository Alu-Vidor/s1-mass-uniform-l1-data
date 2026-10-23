# Source provenance

`original_sha256.json` records the producer-source identities retained by
the study metadata. The manufactured study's `SOURCE_SHA256.json` records
the portable source copies distributed with its archived results.

`source_adaptations.json` identifies the archived manufactured driver
before its output path and report metadata were simplified. The table
extractor checks that identity and verifies the current source copies
against the root `CHECKSUMS.json`. Saved numerical arrays are unchanged.

Per-study producer manifests describe the saved calculations. Use the
root checksum file to check the files in this repository.
