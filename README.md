# Blueprint for Building Autoregressive TTS
We will be teaching an LLM to generate audio tokens in this repository. Consider this as a blueprint for training autoregressive Text-to-Speech models.

## What We're Actually Doing Here

Let's start with the basics. What is a token? It's just a number. That's all it is.

When you type "hello world" into ChatGPT, the model doesn't actually see the words "hello" and "world". It sees numbers. Maybe `[31373, 995]`. The tokenizer converts your text into numbers, the model predicts the next number, and then the detokenizer converts those numbers back into text that you can read.

**Example:**
```
Text:   "The cat sat on the mat"
Tokens: [464, 3797, 3332, 319, 262, 2603]
```

The model never sees "cat" or "mat". It just sees `3797` and `2603`. It learns that after `[464, 3797, 3332, 319, 262]`, the next number is probably `2603`. That's language modeling in a nutshell.

**So here's the insight:** If models just predict numbers, why can't we teach them to predict numbers that represent audio?

That's exactly what we're doing. We're teaching an LLM to generate a few more numbers. Some of those numbers represent words, and some represent audio. The model doesn't know the difference. It just learns: "when I see these numbers, the next number should be this."

## How Do You Get Audio Tokens?

For text, you use a tokenizer that comes with the model. BERT has a tokenizer, GPT has a tokenizer, Qwen has a tokenizer. They all convert text to numbers.

Current LLMs doens't often come with audio tokenizers, therefor we need to find something that does. There's comes **audio codecs** that takes audio and returns numbers. SNAC is one of the most used audio codecs out there currenlty.

**Example with SNAC codec:**

The audio codec compresses the continuous waveform into discrete tokens. Just like a text tokenizer compresses "hello" into `31373`, the audio codec compresses sound into numbers. Something like this:

```
Audio:  [0.02, 0.04, 0.01, -0.03, ...]  (waveform samples)
        ↓ SNAC Encoder
Tokens: [2847, 1092, 3856, 457, 2901, 3345, 1823, ...]  (audio tokens)
```

## Understanding Audio Codecs

Every audio codec has its own way of generating tokens. Some use one number per audio frame. Some use multiple numbers. Some arrange them in layers. You need to understand your codec's pattern to use it properly.

**SNAC's pattern:**

SNAC uses something called a **hierarchical codebook**. Instead of one number per frame, it gives you multiple numbers arranged in 3 layers.

Think of it like describing a photo at different levels of detail:
- **Layer 0 (coarse):** "It's a cat" → captures the main subject
- **Layer 1 (medium):** "It's an orange tabby cat" → captures more details  
- **Layer 2 (fine):** "It's an orange tabby cat with green eyes and a white patch on its nose" → captures fine details

SNAC does the same for audio:
- **Layer 0 (1 token per frame):** What phoneme is this? Is the pitch high or low?
- **Layer 1 (2 tokens per frame):** What does the speaker sound like? Male or female voice?
- **Layer 2 (4 tokens per frame):** What's the breathiness? Any background noise?

So for each audio frame, you get **7 total tokens**: 1 from Layer 0, 2 from Layer 1, and 4 from Layer 2.

**Example for one audio frame:**
```
Audio frame: "ah" sound (40ms of audio)
Layer 0: [2847]                    # phoneme + pitch
Layer 1: [1092, 3856]              # speaker characteristics
Layer 2: [457, 2901, 3345, 1823]   # fine details

All together: [2847, 1092, 3856, 457, 2901, 3345, 1823]
```

Each layer uses a "codebook" of 4096 possible values. Layer 0 tokens can be 0-4095. Layer 1 tokens can be 0-4095. Layer 2 tokens can be 0-4095. That's why it's called **vector quantization** (VQ). The encoder looks up the closest match in its codebook and returns that code number.

## Teaching the LLM the Pattern

LLMs are, in essence, next-token predictors. You give them a sequence of numbers, they predict the next number. That's literally all they do. **Once you have the tokens, it's all the same for the LLM.** Once trained, the model will generate these tokens, no questions asked. The model doesn't care if token `31373` means "hello" and token `151936` means "an 'ah' sound at 440 Hz". It just learns the patterns.

So what we have to do is arrange those audio tokens in order and teach the LLM to learn the pattern. What token comes after what for what sound.

**The training data looks like this:**
```
Text tokens:  [31373, 995]           # "hello world"
Audio start:  [164226]                # <audio_start> special token
Audio tokens: [2847, 1092, 3856, ...] # the actual audio
Audio end:    [164227]                # <audio_end> special token

Combined sequence:
[31373, 995, 164226, 2847, 1092, 3856, 457, 2901, 3345, 1823, ..., 164227]
 └─text──┘  └start┘ └──────────────audio tokens──────────────┘  └─end─┘
```

The LLM learns: "When I see text tokens followed by the start token, I should generate audio tokens that correspond to those words."

During training:
- Input: `[31373, 995, 164226, 2847, 1092, 3856, ...]`
- Model predicts next token: `457`
- Input: `[31373, 995, 164226, 2847, 1092, 3856, 457, ...]`
- Model predicts next token: `2901`
- And so on...

It's learning the statistical pattern of audio tokens just like it learned the statistical pattern of text tokens.

## The Complete Picture

Let me show you the entire flow from text to speech:

**1. Training Phase**

You have a dataset of text-audio pairs. Like `("hello world", hello_world.wav)`.

```
Step 1: Tokenize the text
   "hello world" → [31373, 995]

Step 2: Encode the audio with SNAC
   hello_world.wav → [2847, 1092, 3856, 457, ..., 1234, 5678]
   (Let's say this is 14 tokens total, which is 2 audio frames)

Step 3: Create the training sequence
   [31373, 995, <start>, 2847, 1092, 3856, 457, 2901, 3345, 1823, 
    3042, 1847, 2938, 4527, 1234, 5678, <end>]
   
Step 4: Train the LLM
   - Mask the text tokens (we don't want to predict text from text)
   - Train it to predict audio tokens after seeing text + <start>
   - Loss is only computed on the audio tokens
```

**2. Inference Phase**

You want to generate speech for new text. Like `"this is a test"`.

```
Step 1: Tokenize the text
   "this is a test" → [5661, 318, 257, 1332]

Step 2: Append the start token
   [5661, 318, 257, 1332, <start>]

Step 3: Generate audio tokens autoregressively
   Model predicts: 1847
   Append: [5661, 318, 257, 1332, <start>, 1847]
   
   Model predicts: 2934
   Append: [5661, 318, 257, 1332, <start>, 1847, 2934]
   
   ... keep going until the model generates <end> or you hit max length
   
   Final: [5661, 318, 257, 1332, <start>, 1847, 2934, 3845, ..., <end>]

Step 4: Extract just the audio tokens
   [1847, 2934, 3845, 2901, 3764, 1092, ...]

Step 5: Decode with SNAC
   Audio tokens → waveform → save as audio file
```

That's it. That's the entire system.

## The Core Components

Now that you understand the intuition, let's break down the actual components you need to build this.

### 1. Audio Codec (The Audio Tokenizer)

You need a neural audio codec that can:
- **Encode:** audio waveform → discrete tokens
- **Decode:** discrete tokens → audio waveform

**Popular options:**
- **SNAC** (what we use) - 7 tokens/frame, 24kHz, hierarchical
- **EnCodec** (Meta) - 8 codebooks, 24kHz, very popular
- **SoundStream** (Google) - similar to EnCodec
- **DAC** (Descript) - high quality, good for music

**How SNAC works under the hood:**

It's a VQ-VAE (Vector Quantized Variational AutoEncoder). Think of it like this:

```
Encoder:
  Audio → Neural Network → Continuous features → [quantize] → Discrete codes
  
  "Quantize" means: look up the closest match in a codebook
  Codebook is just a table of 4096 learned vectors
  You find the closest vector, return its index (0-4095)

Decoder:
  Discrete codes → [lookup in codebook] → Continuous features → Neural Network → Audio
```

The encoder and decoder are trained together. During training, SNAC learns:
- What 4096 vectors to store in each codebook (Layer 0, 1, 2)
- How to encode audio into these codes
- How to decode codes back to audio that sounds like the original

Once trained, the codebook is frozen. You just use the encoder to get codes, and the decoder to get audio back.

### 2. Vocabulary Extension

Here's a practical problem: your LLM has a vocabulary of, say, 152,000 tokens (for text). SNAC gives you audio tokens in the range 0-4095 for each layer.

You need to make sure audio tokens don't overlap with text tokens. Otherwise, the model gets confused. Is token `1000` the word "the" or an audio code?

**Solution: Offset the audio tokens**

```
Text tokens: 0 to 151,935        (Qwen's original vocabulary)
Audio tokens: 151,936 to 164,225 (12,290 new tokens we add)

Layer 0 codes (0-4095)   → Add offset 151,936 → IDs 151,936 to 156,031
Layer 1 codes (0-4095)   → Add offset 156,032 → IDs 156,032 to 160,127
Layer 2 codes (0-4095)   → Add offset 160,128 → IDs 160,128 to 164,223
Special tokens           → IDs 164,224 to 164,225 (<audio_start>, <audio_end>)
```

Now every token has a unique ID. The model can distinguish "this is the word 'hello'" from "this is the start of an audio sequence."

**In code:**
```python
# Extend the tokenizer
new_tokens = []
for layer in range(3):
    for code in range(4096):
        new_tokens.append(f"<snac_l{layer}_{code}>")
new_tokens += ["<audio_start>", "<audio_end>"]

tokenizer.add_tokens(new_tokens)  # Adds 12,290 tokens
model.resize_token_embeddings(len(tokenizer))  # Resize model to handle new vocab
```

Now the model has 164,226 tokens total. It can predict both text and audio.

### 3. Flattening the Hierarchy

SNAC gives you hierarchical codes:
```
Layer 0: [c0_0, c0_1, c0_2, ...]        # 1 per frame
Layer 1: [c1_0, c1_1, c1_2, c1_3, ...]  # 2 per frame
Layer 2: [c2_0, c2_1, ..., c2_7, ...]   # 4 per frame
```

LLMs need a flat, 1D sequence. So we flatten using a fixed pattern.

**For each frame, we arrange tokens like this:**
```
Frame 0: [c0_0, c1_0, c1_1, c2_0, c2_1, c2_2, c2_3]
Frame 1: [c0_1, c1_2, c1_3, c2_4, c2_5, c2_6, c2_7]
Frame 2: [c0_2, c1_4, c1_5, c2_8, c2_9, c2_10, c2_11]
...
```

Every 7 tokens = 1 audio frame. The pattern is deterministic, so we can reverse it during inference.

This flattening approach is borrowed from [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS), which demonstrated that this hierarchical pattern works well for autoregressive TTS. See [Flattening Method](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb#scrollTo=zK94B-Pfioto) for the implementation reference. Our `prepare_dataset.py` follows the same principle.


### 4. Training

Once you have the flat sequence of tokens, training is straightforward. You use standard causal language modeling.

**Loss masking:**
```
Input:  [text_tokens, <start>, audio_tokens]
Labels: [-100, -100,   -100,    audio_tokens, <end>]
```

The `-100` tells the model to ignore those positions during loss calculation. We only compute loss on the audio tokens. This way, the model learns:
- TEXT → AUDIO mapping
- Not TEXT → TEXT (it already knows that from pretraining)


### 5. Inference and Decoding

During inference, you reverse everything.

**Steps:**
```
1. Tokenize text: "hello" → [31373]
2. Add start: [31373, <start>]
3. Generate: Model autoregressively predicts audio tokens
   → [31373, <start>, 2847, 1092, 3856, 457, 2901, 3345, 1823, ..., <end>]
4. Extract audio tokens: [2847, 1092, 3856, 457, 2901, 3345, 1823, ...]
5. Unflatten to hierarchical codes:
   Frame 0: L0=[2847], L1=[1092, 3856], L2=[457, 2901, 3345, 1823]
6. Decode with SNAC: hierarchical codes → waveform
7. Save: output.wav
```

**Unflattening code:**
```python
def unflatten(flat_tokens):
    c0, c1, c2 = [], [], []
    for i in range(0, len(flat_tokens), 7):  # Every 7 tokens is 1 frame
        frame = flat_tokens[i:i+7]
        c0.append(frame[0])           # 1st token → Layer 0
        c1.extend(frame[1:3])         # 2nd-3rd tokens → Layer 1
        c2.extend(frame[3:7])         # 4th-7th tokens → Layer 2
    return [c0, c1, c2]
```

Then you pass `[c0, c1, c2]` to SNAC's decoder, and it reconstructs the audio.


---
---

Enough theory. Let's actually build this thing.
## How to Use This Pipeline
### Requirements

- **GPU:** CUDA-capable with tons of VRAM (tested on RTX 3060)
- **Python:** 3.12 or higher
- **Dataset:** LJSpeech-1.1 (free, ~2.6GB)
- **Time:** ~ Depens on the size of your pocket

### Installation

```bash
git clone https://github.com/asiff00/TTS-TRAINING-PIPELINE.git
cd TTS-TRAINING-PIPELINE

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Step 1: Download Dataset (Just once)

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

You get a folder `LJSpeech-1.1/` with:
- `metadata.csv` (text transcripts)
- `wavs/` (13,100 audio files)

Single speaker, clean English speech, ~24 hours total.

### Step 2: Prepare Dataset

This script does the audio encoding and flattening.

```bash
python prepare_dataset.py
```

### Step 3: Train the Model

```bash
python train.py
```

Monitor at [wandb.ai](https://wandb.ai). Best checkpoint is auto-saved.

### Step 4: Generate Speech

```bash
python infer.py
```

Output: `output.wav` (24kHz audio)

## Adapting This Blueprint

Want to customize? Here's what to change.

### Use a Different LLM

In `train.py`:
```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

Any causal LM works. Bigger models → better quality but slower training.

### Use a Different Codec

In `prepare_dataset.py` and `infer.py`:
```python
from encodec import EncodecModel
codec = EncodecModel.encodec_model_24khz()
```

You'll need to adjust flattening based on codec structure. EnCodec uses 8 codebooks instead of 3 layers.

### Use a Different Dataset

In `prepare_dataset.py`:
```python
# Load your custom dataset
metadata = pd.read_csv("my_dataset.csv")
# Expected format: columns for 'audio_path' and 'text'
```

Any speech dataset works. Multi-speaker is fine too (model learns multiple voices).

### Train Longer

In `train.py`:
```python
max_train_steps = 15000  # or 50000 for production quality
```

More steps = better quality, but watch for overfitting. Monitor eval loss.


## What Other Models Do

Other teams have built on these same principles:

**[Maya1](https://huggingface.co/maya-research/maya1) (Maya Research, November 2025):** Production-ready expressive TTS with 3B parameter decoder-only transformer. Uses SNAC codec. Supports 20+ emotion tags (laugh, cry, whisper, rage). Natural language voice control. Sub-100ms latency. Open source under Apache 2.0.

**[Orpheus](https://github.com/canopylabs/orpheus-tts) (Canopy Labs, 2025):** Single-stage autoregressive TTS. 3B parameter medium model released in 2025, with multilingual support added in April 2025. Clean implementation that inspired much of this pipeline.

**[NeuTTS Air](https://huggingface.co/neuphonic/NeuTTS-Air) (Neuphonic, October 2025):** On-device TTS with 0.5B Qwen backbone. Instant voice cloning from 3-second reference. Runs locally on phones, laptops, Raspberry Pi without cloud dependency.

**[Moshi](https://github.com/kyutai-labs/moshi) (Kyutai, September 2024):** Real-time full-duplex speech using Mimi codec. Released September 17, 2024. Can listen and speak simultaneously with ~200ms latency. Revolutionary for conversational AI.

**[Bark](https://github.com/suno-ai/bark) (Suno AI, April 2023):** Released April 20, 2023. Fully open source. Three-stage cascade supporting multiple languages, emotions, and non-speech sounds (laughter, music). Pioneer in codec-based LM TTS.

*All follow the same blueprint: audio codec → flatten to tokens → train LLM → generate → unflatten → decode. They differ in codec choice, number of stages, and scale of training data.*

## Acknowledgements

This implementation was inspired by and borrows techniques from:
- [Orpheus](https://github.com/canopyai/Orpheus-TTS) - Training methodology and architecture patterns
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - Base language model
- [SNAC](https://github.com/hubertsiuzdak/snac) - Audio codec
- [Ray](https://github.com/ray-project/ray) - Distributed training framework
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) - Speech dataset