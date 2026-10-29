import os
import sys
import shutil
import time
import requests

URL = "https://huggingface.co/Amrrs/wav2vec2-large-xlsr-53-tamil/resolve/main/pytorch_model.bin"
TOTAL_SIZE = 1262147031
BLOB_DIR = r"C:\Users\Dhanu Ragavi\.cache\huggingface\hub\models--Amrrs--wav2vec2-large-xlsr-53-tamil\blobs"
SNAPSHOT_DIR = r"C:\Users\Dhanu Ragavi\.cache\huggingface\hub\models--Amrrs--wav2vec2-large-xlsr-53-tamil\snapshots\017e2f0cf4b196d1656e310c87403fe446e39581"
INCOMPLETE_PATH = os.path.join(BLOB_DIR, "06846fb6fa11965e4dffb916e4d0f36c48c2ca12eb9b8387dadb4829ba49cf66.incomplete")
FINAL_BLOB_PATH = os.path.join(BLOB_DIR, "06846fb6fa11965e4dffb916e4d0f36c48c2ca12eb9b8387dadb4829ba49cf66")
FINAL_SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "pytorch_model.bin")

def download_resumable():
    existing_bytes = 0
    if os.path.exists(INCOMPLETE_PATH):
        existing_bytes = os.path.getsize(INCOMPLETE_PATH)
    
    print(f"Resuming download from byte {existing_bytes} ({existing_bytes/(1024*1024):.1f} MB / {TOTAL_SIZE/(1024*1024):.1f} MB)...", flush=True)
    
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
    
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(INCOMPLETE_PATH):
                existing_bytes = os.path.getsize(INCOMPLETE_PATH)
            headers["Range"] = f"bytes={existing_bytes}-"
            
            if existing_bytes >= TOTAL_SIZE:
                print("Download already complete!", flush=True)
                break
                
            print(f"[Attempt {attempt}] Connecting with Range: {headers.get('Range', 'all')}...", flush=True)
            r = requests.get(URL, headers=headers, stream=True, timeout=30)
            
            if r.status_code not in (200, 206):
                print(f"Unexpected status {r.status_code}, retrying...", flush=True)
                time.sleep(3)
                continue
                
            mode = "ab" if existing_bytes > 0 and r.status_code == 206 else "wb"
            if mode == "wb":
                existing_bytes = 0
                
            with open(INCOMPLETE_PATH, mode) as f:
                last_print = time.time()
                bytes_since_print = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    existing_bytes += len(chunk)
                    bytes_since_print += len(chunk)
                    
                    if time.time() - last_print >= 5:
                        speed_mb = (bytes_since_print / (1024 * 1024)) / (time.time() - last_print)
                        pct = (existing_bytes / TOTAL_SIZE) * 100
                        print(f"[{pct:.1f}%] {existing_bytes/(1024*1024):.1f} MB / {TOTAL_SIZE/(1024*1024):.1f} MB ({speed_mb:.2f} MB/s)", flush=True)
                        last_print = time.time()
                        bytes_since_print = 0
                        
            if existing_bytes >= TOTAL_SIZE:
                print(f"Download complete: {existing_bytes} bytes!", flush=True)
                break
        except Exception as e:
            print(f"Connection notice: {e}. Retrying in 2 seconds...", flush=True)
            time.sleep(2)
            
    if os.path.exists(INCOMPLETE_PATH) and os.path.getsize(INCOMPLETE_PATH) >= TOTAL_SIZE:
        print("Finalizing cache files...", flush=True)
        if os.path.exists(FINAL_BLOB_PATH):
            os.remove(FINAL_BLOB_PATH)
        shutil.move(INCOMPLETE_PATH, FINAL_BLOB_PATH)
        
        # Link or copy to snapshot directory
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        if os.path.exists(FINAL_SNAPSHOT_PATH):
            os.remove(FINAL_SNAPSHOT_PATH)
        try:
            os.link(FINAL_BLOB_PATH, FINAL_SNAPSHOT_PATH)
            print("Created hardlink to snapshots folder.", flush=True)
        except Exception:
            shutil.copyfile(FINAL_BLOB_PATH, FINAL_SNAPSHOT_PATH)
            print("Copied model file to snapshots folder.", flush=True)
        print("WAV2VEC2 MODEL CACHE READY!", flush=True)
        return True
    return False

if __name__ == "__main__":
    download_resumable()
