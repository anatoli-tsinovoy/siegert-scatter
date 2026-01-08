#!/usr/bin/env python3
"""
Publish release artifacts to Zenodo via the REST API.

This script handles both first-run bootstrapping (no concept ID) and subsequent
releases (new version of existing concept).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


def load_zenodo_metadata() -> dict:
    """Load metadata from .zenodo.json."""
    zenodo_json = Path(".zenodo.json")
    if not zenodo_json.exists():
        print("ERROR: .zenodo.json not found", file=sys.stderr)
        sys.exit(1)
    with zenodo_json.open() as f:
        return json.load(f)


def get_latest_record(base_url: str, token: str, conceptrecid: str) -> dict | None:
    """Query Zenodo for the latest record in a concept."""
    resp = requests.get(
        f"{base_url}/records/",
        params={
            "q": f"conceptrecid:{conceptrecid}",
            "sort": "mostrecent",
            "size": "1",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    return hits[0] if hits else None


def create_new_version(base_url: str, token: str, record_id: int) -> dict:
    """Create a new version draft from an existing record."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/newversion",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Follow latest_draft link to get the new deposition
    latest_draft_url = data["links"]["latest_draft"]
    resp = requests.get(
        latest_draft_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def create_new_deposition(base_url: str, token: str) -> dict:
    """Create a brand new deposition (first-run bootstrap)."""
    resp = requests.post(
        f"{base_url}/deposit/depositions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def clear_existing_files(base_url: str, token: str, deposition_id: int) -> None:
    """Remove all existing files from a deposition draft."""
    resp = requests.get(
        f"{base_url}/deposit/depositions/{deposition_id}/files",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    for file_info in resp.json():
        file_id = file_info["id"]
        del_resp = requests.delete(
            f"{base_url}/deposit/depositions/{deposition_id}/files/{file_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        del_resp.raise_for_status()


def upload_file(bucket_url: str, token: str, filepath: Path) -> None:
    """Upload a file to the deposition bucket."""
    with filepath.open("rb") as f:
        resp = requests.put(
            f"{bucket_url}/{filepath.name}",
            data=f,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
        )
    resp.raise_for_status()
    print(f"  Uploaded: {filepath.name}")


def update_metadata(
    base_url: str,
    token: str,
    deposition_id: int,
    metadata: dict,
    version: str,
    tag: str,
    repo: str,
) -> None:
    """Update deposition metadata."""
    # Add dynamic fields
    metadata["version"] = version
    metadata["publication_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Add related identifier for GitHub release
    related_identifiers = metadata.get("related_identifiers", [])
    related_identifiers.append(
        {
            "identifier": f"https://github.com/{repo}/releases/tag/{tag}",
            "relation": "isSupplementTo",
            "scheme": "url",
        }
    )
    metadata["related_identifiers"] = related_identifiers

    resp = requests.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"metadata": metadata},
        timeout=30,
    )
    resp.raise_for_status()


def publish_deposition(base_url: str, token: str, deposition_id: int) -> dict:
    """Publish the deposition."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish to Zenodo")
    parser.add_argument("--base-url", required=True, help="Zenodo API base URL")
    parser.add_argument("--token", required=True, help="Zenodo access token")
    parser.add_argument("--version", required=True, help="Version string (without v)")
    parser.add_argument("--tag", required=True, help="Git tag (with v)")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument("--conceptrecid", help="Existing concept record ID")
    parser.add_argument("files", nargs="+", help="Files to upload")
    args = parser.parse_args()

    # Load base metadata
    metadata = load_zenodo_metadata()

    # Resolve file paths (handle globs passed as arguments)
    files = []
    for pattern in args.files:
        path = Path(pattern)
        if path.exists():
            files.append(path)
        else:
            print(f"WARNING: File not found: {pattern}", file=sys.stderr)

    if not files:
        print("ERROR: No files to upload", file=sys.stderr)
        sys.exit(1)

    print(f"Files to upload: {[f.name for f in files]}")

    # Create or get deposition
    if args.conceptrecid:
        print(f"Looking up latest record for concept {args.conceptrecid}...")
        record = get_latest_record(args.base_url, args.token, args.conceptrecid)
        if record:
            print(f"Creating new version from record {record['id']}...")
            deposition = create_new_version(args.base_url, args.token, record["id"])
            # Clear any existing files from the draft
            clear_existing_files(args.base_url, args.token, deposition["id"])
        else:
            print("WARNING: No existing record found, creating new deposition")
            deposition = create_new_deposition(args.base_url, args.token)
    else:
        print("No concept ID provided, creating new deposition (bootstrap mode)...")
        deposition = create_new_deposition(args.base_url, args.token)

    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    conceptrecid = deposition.get("conceptrecid")

    print(f"Deposition ID: {deposition_id}")
    print(f"Concept Record ID: {conceptrecid}")

    # Upload files
    print("Uploading files...")
    for filepath in files:
        upload_file(bucket_url, args.token, filepath)

    # Update metadata
    print("Updating metadata...")
    update_metadata(
        args.base_url,
        args.token,
        deposition_id,
        metadata,
        args.version,
        args.tag,
        args.repo,
    )

    # Publish
    print("Publishing deposition...")
    published = publish_deposition(args.base_url, args.token, deposition_id)

    # Extract DOI info
    conceptdoi = published.get("conceptdoi", "")
    versiondoi = published.get("doi", "")
    record_url = published.get("links", {}).get("record", "")

    # Output results
    print("\n" + "=" * 60)
    print("ZENODO PUBLISH SUCCESSFUL")
    print("=" * 60)
    print(f"Concept Record ID: {conceptrecid}")
    print(f"Concept DOI: {conceptdoi}")
    print(f"Version DOI: {versiondoi}")
    print(f"Record URL: {record_url}")
    print("=" * 60)

    # Write to file for GitHub Actions
    output = {
        "conceptrecid": conceptrecid,
        "conceptdoi": conceptdoi,
        "versiondoi": versiondoi,
        "record_url": record_url,
    }
    with open("zenodo_output.json", "w") as f:
        json.dump(output, f, indent=2)

    # Write to GitHub Actions step summary if available
    summary_file = Path.home() / ".github_step_summary"
    github_summary = Path(
        __import__("os").environ.get("GITHUB_STEP_SUMMARY", summary_file)
    )
    try:
        with github_summary.open("a") as f:
            f.write("\n## Zenodo Publish Results\n\n")
            f.write(f"- **Concept Record ID**: `{conceptrecid}`\n")
            f.write(f"- **Concept DOI**: `{conceptdoi}`\n")
            f.write(f"- **Version DOI**: `{versiondoi}`\n")
            f.write(f"- **Record URL**: {record_url}\n")
    except (OSError, PermissionError):
        pass  # Not in GitHub Actions or can't write


if __name__ == "__main__":
    main()
