import os
import subprocess
import time
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
# Set your AIHub Dataset Key here or pass via argument
DEFAULT_DATASET_KEY = "" 
DOWNLOAD_DIR = "dataset_downloads"
AIHUBSHELL_PATH = "aihubshell" # Ensure this is in PATH or provide full path

def load_file_keys(list_file):
    if not os.path.exists(list_file):
        print(f"[!] File key list not found: {list_file}")
        return []
    with open(list_file, 'r') as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys

def match_file_exists(download_dir, file_key):
    # Naive check: does a file with this key in its name exist?
    # Actual filename might differ from key, but usually key is related.
    # If uncertain, we rely on aihubshell's internal skip or just overwrite.
    # For now, let's assume we proceed unless specific logic is added.
    return False

def download_file(dataset_key, file_key, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    
    cmd = [
        AIHUBSHELL_PATH,
        "-mode", "d",
        "-datasetkey", dataset_key,
        "-filekey", file_key
    ]
    
    # Note: aihubshell might need a download path argument, 
    # but usually it downloads to current dir or configured dir.
    # We might need to cd into download_dir or move file after.
    
    print(f"[*] Starting download for key: {file_key}")
    try:
        # Execute in the download directory to keep things clean
        result = subprocess.run(cmd, cwd=download_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[+] Success: {file_key}")
            return True
        else:
            print(f"[-] Failed: {file_key}")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[!] Exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AI Hub Selective Downloader")
    parser.add_argument("--dataset-key", "-k", default=DEFAULT_DATASET_KEY, help="AI Hub Dataset Key")
    parser.add_argument("--file-list", "-f", required=True, help="Text file containing one filekey per line")
    parser.add_argument("--out-dir", "-o", default=DOWNLOAD_DIR, help="Download Directory")
    
    args = parser.parse_args()
    
    if not args.dataset_key:
        print("[!] Dataset Key is required. Edit script or pass --dataset-key")
        return

    keys = load_file_keys(args.file_list)
    print(f"[*] Loaded {len(keys)} file keys.")
    
    success_count = 0
    fail_count = 0
    
    for i, key in enumerate(keys):
        print(f"\nProcessing {i+1}/{len(keys)}: {key}")
        if download_file(args.dataset_key, key, args.out_dir):
            success_count += 1
        else:
            fail_count += 1
        
        # Optional: Sleep to prevent rate limiting if strict
        time.sleep(1)

    print(f"\n==============================")
    print(f"Download Complete.")
    print(f"Success: {success_count}")
    print(f"Failed : {fail_count}")
    print(f"==============================")

if __name__ == "__main__":
    main()
