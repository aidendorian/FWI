import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

FOLDERS = {
    "cfa_train": {
        "source":   "1epsIb6QLHcd5UOLTaTEGh3trECFNim9O",
        "velocity": "1epsIb6QLHcd5UOLTaTEGh3trECFNim9O",
    },
    "cfa_test": {
        "source":   "1epsIb6QLHcd5UOLTaTEGh3trECFNim9O",
        "velocity": "1epsIb6QLHcd5UOLTaTEGh3trECFNim9O",
    }
}

SOURCE_FILES   = ["data_x", "data_z"]
VELOCITY_FILES = ["vp", "vs", "pm", "pr"]

OUTPUT_DIRS = {
    "cfa_train": {
        "source":   "data/cfa/source_info",
        "velocity": "data/cfa/velocity_models",
    },
    "cfa_test": {
        "source":   "data/test/cfa/source_info",
        "velocity": "data/test/cfa/velocity_models",
    }
}

# ── Tuning knobs ────────────────────────────────────────────────────────────
MAX_WORKERS  = 6        # concurrent downloads; raise to 8-10 if CPU/RAM allow
CHUNK_SIZE   = 32 * 1024 * 1024   # 32 MB chunks (default is 10 MB)
# ────────────────────────────────────────────────────────────────────────────

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def authenticate():
    creds = service_account.Credentials.from_service_account_file(
        'data/service-account-key.json', scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


def make_service():
    """Each thread needs its own service instance (not thread-safe to share)."""
    creds = service_account.Credentials.from_service_account_file(
        'data/service-account-key.json', scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


def get_file_mapping(service, folder_id):
    mapping = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=1000,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        for f in resp.get('files', []):
            mapping[f['name']] = f['id']
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return mapping


def download_file(file_id, file_name, download_path):
    dest = os.path.join(download_path, file_name)
    tmp  = dest + ".tmp"          # write to .tmp, rename on success
    if os.path.exists(dest):
        safe_print(f"  skip (exists): {file_name}")
        return file_name, True, 0.0

    service = make_service()
    try:
        request = service.files().get_media(fileId=file_id)
        start   = time.time()

        with open(tmp, 'wb') as f:
            dl   = MediaIoBaseDownload(f, request, chunksize=CHUNK_SIZE)
            done = False
            while not done:
                status, done = dl.next_chunk()
                elapsed = time.time() - start
                dl_mb   = status.resumable_progress / 1024 / 1024
                speed   = dl_mb / elapsed if elapsed > 0 else 0
                safe_print(f"  {file_name}: {int(status.progress()*100)}%  {speed:.1f} MB/s", end="\r")

        os.rename(tmp, dest)      # atomic swap — no corrupt files on crash
        elapsed = time.time() - start
        size_mb = os.path.getsize(dest) / 1024 / 1024
        safe_print(f"  ✓ {file_name} ({size_mb:.1f} MB in {elapsed:.1f}s — {size_mb/elapsed:.1f} MB/s)")
        return file_name, True, size_mb

    except Exception as e:
        safe_print(f"  ✗ {file_name}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)        # clean up partial file
        return file_name, False, 0.0


def download(dataset, split, start_index, end_index):
    key = f"{dataset}_{split}"
    if key not in FOLDERS:
        print(f"Unknown dataset/split: {key}. Options: {list(FOLDERS.keys())}")
        sys.exit(1)

    service = authenticate()
    grand_start = time.time()
    total_mb    = 0.0

    for kind, prefixes in [("source", SOURCE_FILES), ("velocity", VELOCITY_FILES)]:
        folder_id  = FOLDERS[key][kind]
        output_dir = OUTPUT_DIRS[key][kind]
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nFetching {kind} file list...")
        mapping = get_file_mapping(service, folder_id)
        print(f"Found {len(mapping)} files in folder.")

        # Build the work queue
        tasks = []
        for i in range(start_index, end_index + 1):
            for prefix in prefixes:
                fname = f"{prefix}_{i}.npy"
                if fname in mapping:
                    tasks.append((mapping[fname], fname, output_dir))
                else:
                    print(f"  ⚠ not found: {fname}")

        print(f"Downloading {len(tasks)} {kind} files with {MAX_WORKERS} workers…")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(download_file, fid, fname, path): fname
                for fid, fname, path in tasks
            }
            for fut in as_completed(futures):
                _, ok, mb = fut.result()
                if ok:
                    total_mb += mb

    elapsed = time.time() - grand_start
    print(f"\n{'─'*50}")
    print(f"Done: {total_mb:.1f} MB in {elapsed:.1f}s — avg {total_mb/elapsed:.1f} MB/s")


def main():
    if len(sys.argv) != 5:
        print("Usage: python download.py <dataset> <split> <start> <end>")
        print("  dataset: cfa | cfb")
        print("  split:   train | test")
        print("  Example: python download.py cfa train 1 96")
        sys.exit(1)

    download(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

if __name__ == "__main__":
    main()
    
