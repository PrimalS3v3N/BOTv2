#!/usr/bin/env python3
"""
GitHub to OneDrive Sync Script

Fetches the latest files from a GitHub repository and syncs them
to a local OneDrive folder. Supports both public and private repos.

Usage:
    python github_onedrive_sync.py <github_url> <onedrive_path> [options]

Examples:
    # Sync a public repo to your OneDrive folder
    python github_onedrive_sync.py https://github.com/owner/repo ~/OneDrive/Backups/repo

    # Sync a specific branch with a GitHub token
    python github_onedrive_sync.py https://github.com/owner/repo ~/OneDrive/Projects/repo \
        --branch develop --token ghp_xxxx

    # Sync via API only (no git required)
    python github_onedrive_sync.py https://github.com/owner/repo ~/OneDrive/Projects/repo \
        --method api

    # Dry run to see what would change
    python github_onedrive_sync.py https://github.com/owner/repo ~/OneDrive/Projects/repo \
        --dry-run
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


# ---------------------------------------------------------------------------
# Sync State
# ---------------------------------------------------------------------------

class SyncState:
    """Tracks what has been synced so unchanged files can be skipped."""

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

    def file_changed(self, rel_path: str, current_hash: str) -> bool:
        prev = self.data.get("files", {}).get(rel_path)
        return prev != current_hash


# ---------------------------------------------------------------------------
# GitHubSync
# ---------------------------------------------------------------------------

class GitHubSync:
    """Sync files from a GitHub repo to a local OneDrive folder."""

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
        self.branch = branch  # None means default branch
        self.exclude = set(exclude or [])
        self.dry_run = dry_run
        self.verbose = verbose

        # API session
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "github-onedrive-sync/1.0"
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"

        self.api_base = "https://api.github.com"
        self.state = SyncState(self.onedrive_path)

    # ---- informational ----

    def _log(self, msg: str):
        print(f"  {msg}")

    def _vlog(self, msg: str):
        if self.verbose:
            print(f"  [verbose] {msg}")

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
        """Get the full recursive tree for a commit."""
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
        """Download raw file content from the repo."""
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

    # ---- sync via git clone/pull ----

    def sync_via_git(self):
        """Clone or pull the repo, then mirror files to OneDrive."""
        print(f"\n[git] Syncing {self.owner}/{self.repo} -> {self.onedrive_path}")

        branch = self.branch or self.get_default_branch()
        self.branch = branch
        print(f"[git] Branch: {branch}")

        with tempfile.TemporaryDirectory(prefix="gh_sync_") as tmp:
            tmp_repo = Path(tmp) / self.repo

            # Clone
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

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=tmp_repo,
            )
            commit_sha = sha_result.stdout.strip()

            if self.state.last_commit == commit_sha:
                print(f"[git] Already up to date (commit {commit_sha[:8]})")
                return

            print(f"[git] New commit: {commit_sha[:8]}")
            if self.state.last_commit:
                print(f"[git] Previous:   {self.state.last_commit[:8]}")

            # Mirror files
            self._mirror_directory(tmp_repo, commit_sha)

        print("[git] Sync complete.\n")

    # ---- sync via API ----

    def sync_via_api(self):
        """Download files via the GitHub API to OneDrive."""
        print(f"\n[api] Syncing {self.owner}/{self.repo} -> {self.onedrive_path}")

        branch = self.branch or self.get_default_branch()
        self.branch = branch
        print(f"[api] Branch: {branch}")

        commit_sha = self.get_latest_commit_sha(branch)

        if self.state.last_commit == commit_sha:
            print(f"[api] Already up to date (commit {commit_sha[:8]})")
            return

        print(f"[api] New commit: {commit_sha[:8]}")
        if self.state.last_commit:
            print(f"[api] Previous:   {self.state.last_commit[:8]}")

        tree = self.get_repo_tree(commit_sha)
        blobs = [item for item in tree if item["type"] == "blob"]
        total = len(blobs)

        if self.dry_run:
            self._log(f"Would sync {total} files (dry run)")
            for item in blobs:
                self._log(f"  {item['path']}")
            return

        self.onedrive_path.mkdir(parents=True, exist_ok=True)

        file_hashes = {}
        synced = 0
        skipped = 0

        for i, item in enumerate(blobs, 1):
            rel_path = item["path"]

            if self._is_excluded(rel_path):
                self._vlog(f"Excluded: {rel_path}")
                skipped += 1
                continue

            dest = self.onedrive_path / rel_path

            # Use git blob SHA as a change indicator
            blob_sha = item["sha"]
            if not self.state.file_changed(rel_path, blob_sha) and dest.exists():
                self._vlog(f"Unchanged: {rel_path}")
                file_hashes[rel_path] = blob_sha
                skipped += 1
                continue

            self._log(f"[{i}/{total}] Downloading {rel_path}")
            try:
                content = self.download_file_content(rel_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
                file_hashes[rel_path] = blob_sha
                synced += 1
            except Exception as e:
                print(f"  WARNING: Failed to download {rel_path}: {e}")

        # Remove files that no longer exist in the repo
        removed = self._cleanup_removed_files(
            {item["path"] for item in blobs}
        )

        self.state.save(commit_sha, file_hashes)
        print(f"[api] Done: {synced} updated, {skipped} unchanged, {removed} removed\n")

    # ---- shared helpers ----

    def _is_excluded(self, rel_path: str) -> bool:
        """Check if a path matches any exclusion pattern."""
        parts = Path(rel_path).parts
        for exc in self.exclude:
            if exc in parts or rel_path.startswith(exc):
                return True
        return False

    def _mirror_directory(self, source_dir: Path, commit_sha: str):
        """Copy files from source_dir to onedrive_path, tracking changes."""
        if self.dry_run:
            self._log("Dry run — listing files that would be synced:")

        self.onedrive_path.mkdir(parents=True, exist_ok=True)

        file_hashes = {}
        synced = 0
        skipped = 0

        # Walk the cloned repo (skip .git directory)
        for src_file in sorted(source_dir.rglob("*")):
            if not src_file.is_file():
                continue

            rel = src_file.relative_to(source_dir)
            rel_str = str(rel)

            # Skip .git internals
            if rel.parts[0] == ".git":
                continue

            if self._is_excluded(rel_str):
                self._vlog(f"Excluded: {rel_str}")
                continue

            current_hash = file_hash(src_file)
            file_hashes[rel_str] = current_hash

            if self.dry_run:
                changed = self.state.file_changed(rel_str, current_hash)
                status = "CHANGED" if changed else "unchanged"
                self._log(f"  [{status}] {rel_str}")
                continue

            dest = self.onedrive_path / rel
            if not self.state.file_changed(rel_str, current_hash) and dest.exists():
                self._vlog(f"Unchanged: {rel_str}")
                skipped += 1
                continue

            self._log(f"Syncing: {rel_str}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)
            synced += 1

        if self.dry_run:
            return

        # Clean up files removed from repo
        removed = self._cleanup_removed_files(set(file_hashes.keys()))

        self.state.save(commit_sha, file_hashes)
        print(f"[git] Done: {synced} updated, {skipped} unchanged, {removed} removed")

    def _cleanup_removed_files(self, current_files: set) -> int:
        """Remove local files that no longer exist in the repo."""
        prev_files = set(self.state.data.get("files", {}).keys())
        to_remove = prev_files - current_files
        removed = 0

        for rel_path in to_remove:
            if self._is_excluded(rel_path):
                continue
            target = self.onedrive_path / rel_path
            if target.exists():
                self._log(f"Removing deleted file: {rel_path}")
                if not self.dry_run:
                    target.unlink()
                    # Remove empty parent directories
                    parent = target.parent
                    while parent != self.onedrive_path:
                        try:
                            parent.rmdir()  # only removes if empty
                            parent = parent.parent
                        except OSError:
                            break
                removed += 1

        return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="github_onedrive_sync",
        description="Sync files from a GitHub repository to a local OneDrive folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s https://github.com/owner/repo ~/OneDrive/Backups/repo
  %(prog)s owner/repo ~/OneDrive/Projects/repo --branch develop
  %(prog)s owner/repo "C:/Users/me/OneDrive/Code/repo" --method api
  %(prog)s owner/repo ~/OneDrive/repo --exclude .github --exclude docs
  %(prog)s owner/repo ~/OneDrive/repo --dry-run
""",
    )
    parser.add_argument(
        "github_url",
        help="GitHub repository URL or owner/repo shorthand",
    )
    parser.add_argument(
        "onedrive_path",
        help="Local OneDrive folder path to sync files into",
    )
    parser.add_argument(
        "--branch", "-b",
        default=None,
        help="Branch to sync (default: repo's default branch)",
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
        help="Sync method: 'git' clones the repo (default), 'api' downloads via GitHub API",
    )
    parser.add_argument(
        "--exclude", "-e",
        action="append",
        default=[],
        help="Exclude files/directories matching this name (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including unchanged files",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        syncer = GitHubSync(
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
    print("  GitHub -> OneDrive Sync")
    print("=" * 60)
    print(f"  Repository : {syncer.owner}/{syncer.repo}")
    print(f"  Destination: {syncer.onedrive_path}")
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
