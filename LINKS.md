# Lab 21 — Links (Submission Option B)

**Học viên**: `<Họ tên>` — `<MSSV>`

## 🔗 GitHub
- Repo code: `https://github.com/<username>/<repo>`

## 🤗 HuggingFace Hub — LoRA adapter
- r=16 (best rank): `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r16`
- *(tùy chọn)* r=8: `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r8`
- *(tùy chọn)* r=64: `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r64`

## 🎁 Stretch artifacts (bonus)
- ALL-layers r=16: `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r16-all`
- DoRA r=16: `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r16-dora`
- GGUF (q4_k_m): `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r16-gguf`
- W&B run: `https://wandb.ai/<user>/lab21-lora-rank`

## ✅ Cách verify (cho instructor)
```python
from peft import PeftModel
from unsloth import FastLanguageModel
base, tok = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-3B-bnb-4bit", load_in_4bit=True)
ft = PeftModel.from_pretrained(base, "<username>/qwen2.5-3b-vi-lab21-r16")
```

---
> Cách push: trong notebook, cell "Push adapter" — đặt `PUSH_TO_HUB = True`, `HUB_REPO_ID = "<username>/qwen2.5-3b-vi-lab21-r16"`,
> và thêm secret `HF_TOKEN` (Colab: 🔑 Secrets ở sidebar trái > New secret, bật "Notebook access"). Sau khi push xong, dán link thật vào file này.
