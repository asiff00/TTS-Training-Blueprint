"""
Generate speech from best checkpoint
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from snac import SNAC
import soundfile as sf

print("[STAGE 1/4] Loading model checkpoint...")
model = AutoModelForCausalLM.from_pretrained("./qwen-snac-tts/checkpoint-best", torch_dtype=torch.float16).cuda()
print("[STAGE 1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_snac")
print("[STAGE 1/4] Loading SNAC decoder...")
decoder = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
SNAC_OFFSET = len(tokenizer) - 12290
print("[STAGE 1/4] All models loaded successfully")

def unflatten(tokens):
    c0, c1, c2 = [], [], []
    i = 0
    while i + 6 < len(tokens):
        f = tokens[i:i+7]
        c0.append(f[0] - SNAC_OFFSET)
        c1.extend([f[1] - SNAC_OFFSET - 4096, f[2] - SNAC_OFFSET - 4096])
        c2.extend([t - SNAC_OFFSET - 8192 for t in f[3:]])
        i += 7
    n = len(c0)
    c1 += [0] * (n*2 - len(c1))
    c2 += [0] * (n*4 - len(c2))
    return [torch.tensor(x, dtype=torch.long).unsqueeze(0).unsqueeze(-1).cuda() for x in [c0, c1, c2]]

print("\n[STAGE 2/4] Waiting for text input...")
text = input("Text → ") or "This is the final working version of open-source TTS on a single RTX 3060."
print(f"[STAGE 2/4] Input text: {text}")

print("[STAGE 3/4] Generating audio tokens...")
prompt = tokenizer(text + "<|audio_start|>", return_tensors="pt").input_ids.cuda()
out = model.generate(prompt, max_new_tokens=1800, do_sample=True, temperature=0.7, top_p=0.9,
                     eos_token_id=tokenizer.convert_tokens_to_ids("<|audio_end|>"))
print("[STAGE 3/4] Audio tokens generated")

start = (out[0] == tokenizer.convert_tokens_to_ids("<|audio_start|>")).nonzero()[0][0] + 1
end = (out[0] == tokenizer.convert_tokens_to_ids("<|audio_end|>")).nonzero()
end = end[0][0] if len(end) else len(out[0])

print("[STAGE 4/4] Decoding to audio...")
codes = unflatten(out[0][start:end].tolist())
audio = decoder.decode(codes).cpu().squeeze().numpy()
print("[STAGE 4/4] Saving audio file...")
sf.write("output.wav", audio, 24000)
print("[STAGE 4/4] ✓ Saved: output.wav")