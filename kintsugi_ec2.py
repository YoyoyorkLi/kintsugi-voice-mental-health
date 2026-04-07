import boto3
import subprocess
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path

# Find dam file at https://huggingface.co/KintsugiHealth/dam
# ── DAM model path (on EC2) ───────────────────────────────────────
sys.path.insert(0, "/data/dam")
from pipeline import Pipeline

# ── CONFIG ────────────────────────────────────────────────────────
BUCKET         = "pandorabioapp5e1f277de78b498e9677dd7be7cb9d09205640-production"
AWS_ACCESS_KEY = "YOUR_ACCESS_KEY"
AWS_SECRET_KEY = "YOUR_SECRET_KEY"
AWS_REGION     = "us-west-2"  

CSV_V1         = "/data/Voice_R1_derived2.csv"
CSV_V2         = "/data/Voice_derived2.csv"
CSV_OUTPUT     = "/data/voice_with_scores.csv"
CHECKPOINT     = 50  # save progress every N rows
# ──────────────────────────────────────────────────────────────────

# Load and join voice data OR load existing progress
if os.path.exists(CSV_OUTPUT):
    print(f"Found existing progress! Loading {CSV_OUTPUT} to resume...")
    voice = pd.read_csv(CSV_OUTPUT)
else:
    print("No checkpoint found. Loading raw data...")
    voice_v1 = pd.read_csv(CSV_V1)
    voice_v2 = pd.read_csv(CSV_V2)
    voice = pd.concat([voice_v1, voice_v2]).reset_index(drop=True)
    
    # Add score columns if starting fresh
    if "depression_score" not in voice.columns:
        voice["depression_score"] = None
    if "anxiety_score" not in voice.columns:
        voice["anxiety_score"] = None
    if "score_status" not in voice.columns:
        voice["score_status"] = None

print(f"Total rows: {len(voice)}")

# Setup AWS
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
s3 = session.client("s3")

# Load model
print("Loading DAM pipeline...")
pipeline = Pipeline()
print("Model ready.\n")

def score_voice_url(voice_url):
    s3_key = "public/" + voice_url
    with tempfile.TemporaryDirectory() as tmpdir:
        aac_path = os.path.join(tmpdir, "audio.aac")
        wav_path = os.path.join(tmpdir, "audio.wav")

        # Download from S3
        try:
            s3.download_file(BUCKET, s3_key, aac_path)
        except Exception as e:
            return None, None, f"s3_error: {str(e)}"

        # Convert AAC → mono WAV at 16kHz
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", aac_path,
                "-ar", "16000", "-ac", "1", wav_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return None, None, "ffmpeg_error"

        # Run model
        try:
            result = pipeline.run_on_file(wav_path, quantize=True)
            return result["depression"], result["anxiety"], "ok"
        except Exception as e:
            return None, None, f"model_error: {str(e)}"

# Only process rows not yet scored
to_process = voice[voice["score_status"].isna()].index
print(f"Rows to process: {len(to_process)}")
print(f"Already scored:  {len(voice) - len(to_process)}")
print("─" * 50)

for i, idx in enumerate(to_process):
    row = voice.loc[idx]
    
    # ── Handle missing (NaN) URLs so the script doesn't crash ──
    if pd.isna(row["voice_url"]):
        print(f"[{i+1}/{len(to_process)}] Row index {idx} has no voice_url. Skipping.")
        voice.at[idx, "score_status"] = "missing_url"
        continue
    # ────────────────────────────────────────────────────────────────

    filename = Path(row["voice_url"]).name
    print(f"[{i+1}/{len(to_process)}] {filename}")

    dep, anx, status = score_voice_url(row["voice_url"])
    voice.at[idx, "depression_score"] = dep
    voice.at[idx, "anxiety_score"]    = anx
    voice.at[idx, "score_status"]     = status

    print(f"  → {status} | depression={dep}, anxiety={anx}")

    # Save checkpoint every N rows
    if (i + 1) % CHECKPOINT == 0:
        voice.to_csv(CSV_OUTPUT, index=False)
        print(f"Checkpoint saved ({i+1} processed in this run)")

# Final save
voice.to_csv(CSV_OUTPUT, index=False)
print(f"\nDone! Results saved to {CSV_OUTPUT}")

# Summary
print("\n── Score Status Summary ──")
print(voice["score_status"].value_counts(dropna=False))