# Lab 21 — Links (Submission Option B)

**Học viên**: `Nguyễn Lý Minh Kỳ` — `2A202600782`

## 🔗 GitHub
- Repo code: `https://github.com/nekloyh/Day21_T3_2A202600782_NguyenLyMinhKy`

## 🤗 HuggingFace Hub — LoRA adapter
- r=16 (best rank): `https://huggingface.co/nekloyh/qwen2.5-3b-vi-lab21-r16`

## 🎁 Stretch artifacts (bonus)
- ALL-layers r=16: `https://huggingface.co/nekloyh/qwen2.5-3b-vi-lab21-r16-all`
- DoRA r=16: `https://huggingface.co/nekloyh/qwen2.5-3b-vi-lab21-r16-dora`
- GGUF (q4_k_m): `https://huggingface.co/nekloyh/qwen2.5-3b-vi-lab21-r16-gguf`
- W&B run: `https://wandb.ai/nglyminhky2k4-nknk/lab21-lora-rank`

## ✅ Cách verify (cho instructor)
```python
from peft import PeftModel
from unsloth import FastLanguageModel
base, tok = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-3B-bnb-4bit", load_in_4bit=True)
ft = PeftModel.from_pretrained(base, "nekloyh/qwen2.5-3b-vi-lab21-r16")
```

---
