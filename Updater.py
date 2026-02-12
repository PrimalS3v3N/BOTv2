#!/usr/bin/env python3
"""
AutoUpdate — GitHub to Local Sync Script

Compares files between a GitHub repository and a local folder,
then updates local copies for files that exist in BOTH locations.
Files unique to either side are left completely untouched.

Usage:
    python Updater.py <github_url> <local_path> [options]

Examples:
    # Update matching files in your local folder from GitHub
    python Updater.py https://github.com/owner/repo ~/Projects/repo

    # Sync a specific branch
    python Updater.py https://github.com/owner/repo ~/Projects/repo \
        --branch develop --token ghp_xxxx

    # Use the API method (no git required)
    python Updater.py owner/repo ~/Projects/repo --method api

    # Preview what would change without writing anything
    python Updater.py owner/repo ~/Projects/repo --dry-run
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required.")
    print("Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYNC_STATE_FILE = ".github_onedrive_sync_state.json"


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_github_url(url: str) -> tuple:
    """
    Extract (owner, repo) from various GitHub URL formats.

    Supported formats:
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/tree/branch
        git@github.com:owner/repo.git
        owner/repo
    """
    url = url.strip().rstrip("/")

    # SSH format
    if url.startswith("git@github.com:"):
        path = url.split("git@github.com:")[-1]
        path = path.removesuffix(".git")
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]

    # HTTPS format
    if "github.com" in url:
        parts = url.split("github.com")[-1].strip("/").split("/")
        if len(parts) >= 2:
            repo = parts[1].removesuffix(".git")
            return parts[0], repo

    # owner/repo shorthand
    if "/" in url and ":" not in url and " " not in url:
        parts = url.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    raise ValueError(
        f"Cannot parse GitHub URL: {url}\n"
        "Expected format: https://github.com/owner/repo  or  owner/repo"
    )


def collect_local_files(directory: Path) -> dict[str, Path]:
    """
    Walk a directory and return {relative_path_str: absolute_path} for every
    file, skipping hidden sync-state files.
    """
    files = {}
    for item in sorted(directory.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(directory)
        rel_str = str(rel)
        # Skip our own state file
        if rel.name == SYNC_STATE_FILE:
            continue
        # Skip hidden .git dirs (shouldn't exist here, but just in case)
        if rel.parts[0] == ".git":
            continue
        files[rel_str] = item
    return files


# ---------------------------------------------------------------------------
# Sync State
# ---------------------------------------------------------------------------

class SyncState:
    """Persists the last-known hashes so unchanged files can be skipped."""

    def __init__(self, onedrive_path: Path):
        self.state_file = onedrive_path / SYNC_STATE_FILE
        self.data = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self, commit_sha: str, file_hashes: dict):
        self.data = {
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "commit_sha": commit_sha,
            "files": file_hashes,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.data, indent=2))

    @property
    def last_commit(self) -> str | None:
        return self.data.get("commit_sha")

    @property
    def last_sync_time(self) -> str | None:
        return self.data.get("last_sync")

    def prev_hash(self, rel_path: str) -> str | None:
        return self.data.get("files", {}).get(rel_path)


# ---------------------------------------------------------------------------
# AutoUpdate
# ---------------------------------------------------------------------------

class AutoUpdate:
    """
    Update OneDrive files from a GitHub repo.

    Only files that already exist in the OneDrive folder AND in the GitHub
    repo are compared and updated.  Files unique to either side are never
    touched or deleted.
    """

    def __init__(
        self,
        github_url: str,
        onedrive_path: str,
        token: str | None = None,
        branch: str | None = None,
        exclude: list | None = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.owner, self.repo = parse_github_url(github_url)
        self.clone_url = f"https://github.com/{self.owner}/{self.repo}.git"
        self.onedrive_path = Path(onedrive_path).expanduser().resolve()
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.branch = branch  # None → will resolve to default branch
        self.exclude = set(exclude or [])
        self.dry_run = dry_run
        self.verbose = verbose

        # API session
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "github-onedrive-sync/2.0"
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"

        self.api_base = "https://api.github.com"
        self.state = SyncState(self.onedrive_path)

    # ---- logging helpers ----

    def _log(self, msg: str):
        print(f"  {msg}")

    def _vlog(self, msg: str):
        if self.verbose:
            print(f"  [verbose] {msg}")

    # ---- exclusion ----

    def _is_excluded(self, rel_path: str) -> bool:
        parts = Path(rel_path).parts
        for exc in self.exclude:
            if exc in parts or rel_path.startswith(exc):
                return True
        return False

    # ---- GitHub API helpers ----

    def _api_get(self, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.api_base}{endpoint}"
        resp = self.session.get(url, **kwargs)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            print("ERROR: GitHub API rate limit reached.")
            print("       Set GITHUB_TOKEN env var or use --token to authenticate.")
            sys.exit(1)
        return resp

    def get_default_branch(self) -> str:
        resp = self._api_get(f"/repos/{self.owner}/{self.repo}")
        resp.raise_for_status()
        return resp.json()["default_branch"]

    def get_latest_commit_sha(self, branch: str) -> str:
        resp = self._api_get(f"/repos/{self.owner}/{self.repo}/commits/{branch}")
        resp.raise_for_status()
        return resp.json()["sha"]

    def get_repo_tree(self, sha: str) -> list:
        resp = self._api_get(
            f"/repos/{self.owner}/{self.repo}/git/trees/{sha}",
            params={"recursive": "1"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("truncated"):
            print("WARNING: Repository tree was truncated (very large repo).")
            print("         Consider using --method git for a complete sync.")
        return data.get("tree", [])

    def download_file_content(self, file_path: str) -> bytes:
        url = (
            f"https://raw.githubusercontent.com/"
            f"{self.owner}/{self.repo}/{self.branch}/{file_path}"
        )
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # sync via git (clone → compare → copy matching)
    # ------------------------------------------------------------------

    def sync_via_git(self):
        print(f"\n[git] Syncing {self.owner}/{self.repo} <-> {self.onedrive_path}")

        branch = self.branch or self.get_default_branch()
        self.branch = branch
        print(f"[git] Branch: {branch}")

        if not self.onedrive_path.exists():
            print(f"ERROR: OneDrive path does not exist: {self.onedrive_path}")
            print("       Create the folder and place the files you want kept in sync.")
            sys.exit(1)

        # Collect local OneDrive files first
        local_files = collect_local_files(self.onedrive_path)
        self._log(f"OneDrive files found: {len(local_files)}")

        with tempfile.TemporaryDirectory(prefix="gh_sync_") as tmp:
            tmp_repo = Path(tmp) / self.repo

            # Shallow clone
            cmd = ["git", "clone", "--depth", "1", "--branch", branch]
            if self.token:
                authed_url = (
                    f"https://x-access-token:{self.token}@github.com/"
                    f"{self.owner}/{self.repo}.git"
                )
                cmd.append(authed_url)
            else:
                cmd.append(self.clone_url)
            cmd.append(str(tmp_repo))

            self._log("Cloning repository...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: git clone failed:\n{result.stderr}")
                sys.exit(1)

            # Commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=tmp_repo,
            )
            commit_sha = sha_result.stdout.strip()
            print(f"[git] Commit: {commit_sha[:8]}")

            # Collect repo files (skip .git/)
            repo_files: dict[str, Path] = {}
            for item in sorted(tmp_repo.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(tmp_repo)
                if rel.parts[0] == ".git":
                    continue
                repo_files[str(rel)] = item

            self._log(f"GitHub files found: {len(repo_files)}")

            # Find common files
            common = set(local_files.keys()) & set(repo_files.keys())
            github_only = set(repo_files.keys()) - set(local_files.keys())
            onedrive_only = set(local_files.keys()) - set(repo_files.keys())

            self._log(f"Files in common:     {len(common)}")
            self._vlog(f"GitHub-only files:   {len(github_only)} (ignored)")
            self._vlog(f"OneDrive-only files: {len(onedrive_only)} (ignored)")

            if self.verbose:
                for f in sorted(github_only):
                    self._vlog(f"  GitHub only  : {f}")
                for f in sorted(onedrive_only):
                    self._vlog(f"  OneDrive only: {f}")

            # Compare and update matching files
            updated = 0
            unchanged = 0
            skipped = 0
            file_hashes = {}

            for rel_path in sorted(common):
                if self._is_excluded(rel_path):
                    self._vlog(f"Excluded: {rel_path}")
                    skipped += 1
                    continue

                src = repo_files[rel_path]
                dest = local_files[rel_path]
                src_hash = file_hash(src)
                dest_hash = file_hash(dest)
                file_hashes[rel_path] = src_hash

                if src_hash == dest_hash:
                    self._vlog(f"Identical: {rel_path}")
                    unchanged += 1
                    continue

                # Content differs — update OneDrive with GitHub version
                if self.dry_run:
                    self._log(f"[WOULD UPDATE] {rel_path}")
                else:
                    self._log(f"Updating: {rel_path}")
                    shutil.copy2(src, dest)
                updated += 1

        if not self.dry_run:
            self.state.save(commit_sha, file_hashes)

        label = "Would update" if self.dry_run else "Updated"
        print(f"[git] Done: {updated} {label.lower()}, "
              f"{unchanged} identical, {skipped} excluded")
        print(f"[git] Skipped {len(github_only)} GitHub-only, "
              f"{len(onedrive_only)} OneDrive-only (untouched)\n")

        return updated

    # ------------------------------------------------------------------
    # sync via API (no git required)
    # ------------------------------------------------------------------

    def sync_via_api(self):
        print(f"\n[api] Syncing {self.owner}/{self.repo} <-> {self.onedrive_path}")

        branch = self.branch or self.get_default_branch()
        self.branch = branch
        print(f"[api] Branch: {branch}")

        if not self.onedrive_path.exists():
            print(f"ERROR: OneDrive path does not exist: {self.onedrive_path}")
            print("       Create the folder and place the files you want kept in sync.")
            sys.exit(1)

        # Collect local OneDrive files
        local_files = collect_local_files(self.onedrive_path)
        self._log(f"OneDrive files found: {len(local_files)}")

        # Get GitHub tree
        commit_sha = self.get_latest_commit_sha(branch)
        print(f"[api] Commit: {commit_sha[:8]}")

        tree = self.get_repo_tree(commit_sha)
        repo_file_set = {
            item["path"]: item["sha"]
            for item in tree
            if item["type"] == "blob"
        }
        self._log(f"GitHub files found: {len(repo_file_set)}")

        # Find common files
        common = set(local_files.keys()) & set(repo_file_set.keys())
        github_only = set(repo_file_set.keys()) - set(local_files.keys())
        onedrive_only = set(local_files.keys()) - set(repo_file_set.keys())

        self._log(f"Files in common:     {len(common)}")
        self._vlog(f"GitHub-only files:   {len(github_only)} (ignored)")
        self._vlog(f"OneDrive-only files: {len(onedrive_only)} (ignored)")

        if self.verbose:
            for f in sorted(github_only):
                self._vlog(f"  GitHub only  : {f}")
            for f in sorted(onedrive_only):
                self._vlog(f"  OneDrive only: {f}")

        # Compare and update matching files
        updated = 0
        unchanged = 0
        skipped = 0
        file_hashes = {}

        for rel_path in sorted(common):
            if self._is_excluded(rel_path):
                self._vlog(f"Excluded: {rel_path}")
                skipped += 1
                continue

            blob_sha = repo_file_set[rel_path]
            prev = self.state.prev_hash(rel_path)
            dest = local_files[rel_path]

            # If the GitHub blob SHA hasn't changed since last sync and the
            # local file hash also matches what we recorded, skip the download.
            if prev == blob_sha:
                dest_hash = file_hash(dest)
                saved_local = self.state.data.get("local_hashes", {}).get(rel_path)
                if saved_local == dest_hash:
                    self._vlog(f"Unchanged: {rel_path}")
                    file_hashes[rel_path] = blob_sha
                    unchanged += 1
                    continue

            # Need to download and compare
            if self.dry_run:
                self._log(f"[WOULD CHECK/UPDATE] {rel_path}")
                updated += 1
                file_hashes[rel_path] = blob_sha
                continue

            try:
                content = self.download_file_content(rel_path)
            except Exception as e:
                print(f"  WARNING: Failed to download {rel_path}: {e}")
                continue

            # Compare downloaded content to local file
            src_hash = hashlib.sha256(content).hexdigest()
            dest_hash = file_hash(dest)

            file_hashes[rel_path] = blob_sha

            if src_hash == dest_hash:
                self._vlog(f"Identical: {rel_path}")
                unchanged += 1
                continue

            self._log(f"Updating: {rel_path}")
            dest.write_bytes(content)
            updated += 1

        if not self.dry_run:
            # Also save local hashes so we can skip re-downloads next time
            local_hashes = {}
            for rel_path in file_hashes:
                p = local_files.get(rel_path)
                if p and p.exists():
                    local_hashes[rel_path] = file_hash(p)
            state_data = {
                "last_sync": datetime.now(timezone.utc).isoformat(),
                "commit_sha": commit_sha,
                "files": file_hashes,
                "local_hashes": local_hashes,
            }
            self.state.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state.state_file.write_text(json.dumps(state_data, indent=2))

        label = "Would update" if self.dry_run else "Updated"
        print(f"[api] Done: {updated} {label.lower()}, "
              f"{unchanged} identical, {skipped} excluded")
        print(f"[api] Skipped {len(github_only)} GitHub-only, "
              f"{len(onedrive_only)} OneDrive-only (untouched)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AutoUpdate",
        description=(
            "Update files in a local OneDrive folder from a GitHub repository. "
            "Only files that exist in BOTH locations are compared and updated. "
            "Files unique to either side are never touched or deleted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s https://github.com/owner/repo ~/OneDrive/Projects/repo
  %(prog)s owner/repo ~/OneDrive/Projects/repo --branch develop
  %(prog)s owner/repo "C:/Users/me/OneDrive/Code/repo" --method api
  %(prog)s owner/repo ~/OneDrive/repo --exclude .github --exclude docs
  %(prog)s owner/repo ~/OneDrive/repo --dry-run --verbose
""",
    )
    parser.add_argument(
        "github_url",
        help="GitHub repository URL or owner/repo shorthand",
    )
    parser.add_argument(
        "onedrive_path",
        help="Local OneDrive folder path containing your files",
    )
    parser.add_argument(
        "--branch", "-b",
        default=None,
        help="Branch to sync from (default: repo's default branch)",
    )
    parser.add_argument(
        "--token", "-t",
        default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["git", "api"],
        default="git",
        help="Sync method: 'git' clones the repo (default), 'api' uses GitHub API",
    )
    parser.add_argument(
        "--exclude", "-e",
        action="append",
        default=[],
        help="Exclude matching files from sync (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output (identical files, ignored files, etc.)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        syncer = AutoUpdate(
            github_url=args.github_url,
            onedrive_path=args.onedrive_path,
            token=args.token,
            branch=args.branch,
            exclude=args.exclude,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 60)
    print("  GitHub <-> OneDrive Sync")
    print("  (update matching files only, nothing deleted)")
    print("=" * 60)
    print(f"  Repository : {syncer.owner}/{syncer.repo}")
    print(f"  OneDrive   : {syncer.onedrive_path}")
    print(f"  Method     : {args.method}")
    if syncer.branch:
        print(f"  Branch     : {syncer.branch}")
    if syncer.token:
        print(f"  Auth       : token (****{syncer.token[-4:]})")
    if syncer.exclude:
        print(f"  Exclude    : {', '.join(syncer.exclude)}")
    if syncer.dry_run:
        print(f"  Mode       : DRY RUN")
    if syncer.state.last_sync_time:
        print(f"  Last sync  : {syncer.state.last_sync_time}")
    print("=" * 60)

    try:
        if args.method == "git":
            syncer.sync_via_git()
        else:
            syncer.sync_via_api()
    except requests.exceptions.ConnectionError:
        print("ERROR: Network connection failed. Check your internet connection.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"ERROR: Repository not found: {syncer.owner}/{syncer.repo}")
            print("       Check the URL or provide a --token for private repos.")
        else:
            print(f"ERROR: HTTP error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSync cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
