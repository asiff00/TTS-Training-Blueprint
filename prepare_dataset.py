"""
Prepare dataset for training
"""
import os
import torch
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from snac import SNAC
from transformers import AutoTokenizer
from datasets import Dataset

print("[STAGE 1/4] Loading SNAC model...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
print("[STAGE 1/4] SNAC model loaded successfully")

print("[STAGE 2/4] Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("[STAGE 2/4] Tokenizer loaded")

print("[STAGE 2/4] Adding SNAC tokens...")
new_tokens = [f"<snac_l{l}_{i}>" for l in range(3) for i in range(4096)] + ["<|audio_start|>", "<|audio_end|>"]
added = tokenizer.add_tokens(new_tokens)
print(f"[STAGE 2/4] Added {added} new tokens → total vocab: {len(tokenizer)}")

print("[STAGE 2/4] Saving and reloading tokenizer for sync...")
os.makedirs("./tokenizer_snac", exist_ok=True)
tokenizer.save_pretrained("./tokenizer_snac")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_snac", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("[STAGE 2/4] Tokenizer synced and saved → ./tokenizer_snac")

def encode_7_per_frame(wav_path):
    audio, sr = sf.read(wav_path)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.no_grad():
        codes = snac_model.encode(audio)
    c0, c1, c2 = [c.squeeze(0).squeeze(-1) for c in codes]
    base_offset = len(tokenizer) - 12290
    flat = []
    for i in range(len(c0)):
        flat.append(base_offset + 0*4096 + c0[i].item())
        flat.append(base_offset + 1*4096 + c1[2*i].item())
        flat.append(base_offset + 1*4096 + c1[2*i+1].item())
        for j in range(4):
            flat.append(base_offset + 2*4096 + c2[4*i+j].item())
    return flat

metadata = pd.read_csv("LJSpeech-1.1/metadata.csv", sep="|", header=None, names=["id", "text", "norm_text"])
data = []
start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

if os.path.exists("lj_speech_snac_ready"):
    print("[STAGE 3/4] Dataset 'lj_speech_snac_ready' already exists. Nothing to do.")
    exit()

print("[STAGE 3/4] Encoding all 13,100 clips...")
for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
    wav_path = f"LJSpeech-1.1/wavs/{row['id']}.wav"
    if not os.path.exists(wav_path):
        continue
    text = row["norm_text"]
    if pd.isna(text) or not isinstance(text, str):
        text = row["text"]
    if pd.isna(text) or not isinstance(text, str):
        print(f"Skipping {row['id']} - no text found")
        continue
    text_ids = tokenizer.encode(text)
    snac_tokens = encode_7_per_frame(wav_path)
    input_ids = text_ids + [start_id] + snac_tokens + [end_id]
    labels = [-100] * (len(text_ids) + 1) + snac_tokens + [end_id]
    data.append({"input_ids": input_ids, "labels": labels})

print("[STAGE 4/4] Creating and saving dataset...")
dataset = Dataset.from_list(data)
dataset.save_to_disk("lj_speech_snac_ready")
print(f"[STAGE 4/4] FINISHED! {len(dataset)} samples ready → lj_speech_snac_ready")