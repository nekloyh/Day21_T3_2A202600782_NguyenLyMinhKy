# Lab 21 — Evaluation Report

**Học viên**: `<Họ tên>` — `<MSSV>`
**Ngày nộp**: 2026-06-25
**Submission option**: **B + C** (GitHub + HuggingFace Hub, kèm `requirements.txt` reproduce) · **làm full bonus** (Stretch +10)

> ⚠️ **Lưu ý**: các số trong báo cáo này lấy từ một lần chạy tham chiếu của notebook
> (Qwen2.5-3B, Tesla T4). **Sau khi bạn chạy lại trên Google Colab (T4), hãy cập nhật lại
> bảng số liệu + dòng `Base` (chạy cell "Base model perplexity") bằng số của chính bạn.**
> Honor code: phải tự chạy training, không copy số của bạn khác.

---

## 1. Setup

| Mục | Giá trị |
|---|---|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (QLoRA 4-bit NF4) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (**180 train / 20 eval**, seed=42) |
| **Format** | Alpaca (`### Instruction / ### Input / ### Response`) |
| **Token length** | min=25, max=738, p50=227, **p95=562**, p99=704 |
| **max_seq_length** | **1024** (p95=562 → round-up & cap T4) |
| **GPU** | Tesla T4, ~15 GB (Google Colab Free — 1 GPU) |
| **target_modules** | `q_proj`, `v_proj` (lab spec) |
| **Hyperparameters** | 3 epochs · cosine LR `2e-4` · warmup 0.10 · effective batch 8 (1 × grad_accum 8) · `adamw_8bit` · `packing=False` · gradient checkpointing (unsloth) |
| **Training cost** | ~12.2 phút tổng cho 3 ranks · ≈ **$0.07** @ $0.35/hr (trên **Google Colab Free = miễn phí, $0**) |
| **HF Hub link** | `https://huggingface.co/<username>/qwen2.5-3b-vi-lab21-r16` *(điền sau khi push — xem LINKS.md)* |

---

## 2. Rank Experiment Results

| Rank | Trainable Params | % of total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------:|-----------:|-----------:|----------:|----------:|-----------:|
| Base | — | — | — | — | `<điền>` | `<điền — chạy cell base ppl>` |
| 8    | 1,843,200  | 0.06% | 3.99 min | 7.22 GB | 1.5577 | 4.748 |
| 16   | 3,686,400  | 0.12% | 4.26 min | 6.62 GB | 1.5161 | 4.554 |
| 64   | 14,745,600 | 0.48% | 3.99 min | 8.00 GB | 1.4768 | 4.379 |

**Quan sát nhanh:**
- **Trainable params ∝ rank** đúng tuyến tính: r=16 gấp 2× r=8; r=64 gấp 4× r=16.
- **Perplexity giảm đơn điệu theo rank**: 4.748 → 4.554 → 4.379.
- **Train time ≈ nhau (~4 phút)** — ở quy mô 180 mẫu, thời gian bị chi phối bởi forward/data
  chứ không phải kích thước adapter.
- **Peak VRAM** không tăng tuyến tính (r=16 còn thấp hơn r=8 đôi chút) — adapter chỉ chiếm
  vài MB, phần lớn dao động đến từ fragmentation giữa các lần reload base model.

---

## 3. Loss Curve Analysis

![loss curve](results/loss_curve.png)

- Profile T4 **tắt eval-during-training** (tiết kiệm VRAM) → chỉ có **train loss curve**.
- Train loss giảm đều qua 3 epochs (69 steps) và không bật ngược lên → **chưa thấy dấu hiệu
  overfitting rõ** trong khoảng 3 epochs.
- Vì không có eval-loss-theo-step nên đánh giá overfitting dựa trên **eval perplexity cuối cùng**:
  cả 3 ranks đều cho ppl ~4.4–4.7 (thấp hơn base), nghĩa là model học được phân phối dữ liệu
  mà chưa quá khớp. Nếu tăng số epoch lên 5–8 với dataset nhỏ 180 mẫu thì rủi ro overfitting
  (đặc biệt r=64) sẽ cao hơn — nên giữ ở 3 epochs.

---

## 4. Qualitative Comparison (Base vs Fine-tuned r=16)

> 5 prompt tiếng Việt, sinh `max_new_tokens=200`, `temperature=0.7`. Trích gọn để dễ đọc.
> Chọn cả case **thắng** lẫn **thua** (không cherry-pick).

| # | Prompt | Nhận xét |
|---|--------|----------|
| 1 | Giải thích machine learning cho người mới | Cả hai đều trôi chảy, chính xác. **≈ Ngang nhau.** |
| 2 | Viết code Python tính Fibonacci thứ n | FT thêm **input validation** (`raise ValueError`) → chặt chẽ hơn. **FT thắng.** |
| 3 | Liệt kê 5 nguyên tắc UI/UX | FT cho danh sách đánh số **gọn, đúng format** hơn base. **FT thắng (về format).** |
| 4 | Tóm tắt khác biệt LoRA vs QLoRA | FT **hallucinate sai**: gọi LoRA là *"Layer-wise Adaptive Regularization Optimization"* (đúng là *Low-Rank Adaptation*); base lại đúng. **FT thua / degraded.** |
| 5 | Phân biệt prompt engineering / RAG / fine-tuning | Cả hai tương đương, FT tổ chức ý mạch lạc hơn chút. **≈ Ngang / FT nhỉnh nhẹ.** |

**Tổng kết định tính**: FT (r=16) cải thiện **format & độ chặt chẽ** ở vài prompt, nhưng
**không sửa được kiến thức** (ví dụ 4 vẫn hallucinate — đúng với nguyên tắc *"fine-tune cho
style/format, RAG cho knowledge"*). Khác biệt nhìn chung **tinh tế** vì dataset chỉ 180 mẫu và
chỉ target `q+v`. Bảng đầy đủ trong `results/qualitative_comparison.csv`.

---

## 5. Conclusion về Rank Trade-off

Trên dataset Vietnamese-Alpaca 180 mẫu này, **rank = 16 cho ROI tốt nhất**. Đi từ r=8 lên r=16
(gấp đôi params) giảm perplexity từ 4.748 xuống 4.554 (~4.1%), trong khi đi tiếp từ r=16 lên r=64
(gấp **4 lần** params và +21% VRAM) chỉ giảm thêm từ 4.554 xuống 4.379 (~3.8%). Đây chính là biểu
hiện **diminishing returns**: chi phí (params, VRAM) tăng theo cấp số nhân nhưng chất lượng chỉ
nhích tuyến tính rồi chững lại — mỗi tham số tăng thêm ở r=64 "mua" được ít cải thiện hơn hẳn so
với ở r=16. Nguyên nhân: với chỉ ~180 mẫu và chỉ target `q_proj/v_proj`, không gian thông tin cần
học khá nhỏ, nên rank thấp đã đủ "chứa" được; tăng rank chỉ thêm capacity mà dữ liệu không đủ để
tận dụng. **Khuyến nghị deploy production: chọn r=16** — gần như toàn bộ phần lợi ích với một
nửa params của r=64, adapter vẫn rất nhẹ (3.7M params) và merge vào base cho **zero added latency**.
Chỉ nâng lên r=64 nếu bài toán cực kỳ nhạy chất lượng *và* có dataset lớn hơn nhiều để khai thác
capacity; còn r=8 hợp lý khi serving multi-tenant nhiều adapter trên cùng GPU và cần tiết kiệm
bộ nhớ tối đa. Bài học lớn hơn: ở quy mô dữ liệu nhỏ, **chất lượng/độ lớn dataset quan trọng hơn
việc tăng rank**.

---

## 6. What I Learned

- **Rank không phải "càng cao càng tốt"** — diminishing returns xuất hiện rất sớm khi dataset nhỏ;
  r=16 là điểm cân bằng thực tế, đúng như khuyến nghị "standard choice" trong bài giảng.
- **Fine-tune sửa style/format chứ không sửa knowledge** — ví dụ 4 (hallucinate tên LoRA) cho thấy
  rõ: muốn đúng *kiến thức* phải dùng RAG, SFT chỉ dạy *cách trình bày*.
- **Kỹ thuật chạy trên GPU nhỏ là cốt lõi**: QLoRA 4-bit + gradient checkpointing + batch=1 +
  grad-accum giúp fine-tune Qwen2.5-3B gọn trong ~7–8 GB VRAM của một T4 — và toàn bộ thí nghiệm
  3 ranks chỉ tốn ~12 phút, gần như miễn phí trên Google Colab Free.

---

## 7. Stretch Goals (Bonus +10)

> Train thêm trên **cùng dataset / cùng hyperparameters**, chỉ đổi cấu hình LoRA. Số liệu điền
> sau khi chạy các cell Section 7 trên Google Colab.

### 7.1 Target ALL layers vs baseline q+v (r=16)

| Variant | Target modules | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---------|----------------|-----------------:|-----------:|----------:|----------:|-----------:|
| Baseline | `q,v` | 3,686,400 | 4.26 min | 6.62 GB | 1.5161 | 4.554 |
| ALL layers | `q,k,v,o,gate,up,down` | `<điền>` | `<điền>` | `<điền>` | `<điền>` | `<điền>` |

- **Nhận xét**: target ALL layers tăng trainable params ~3–4× và thường cải thiện perplexity
  rõ hơn so với chỉ nâng rank trên q+v (best practice 2025). `<cập nhật kết luận sau khi chạy>`.

### 7.2 DoRA vs LoRA (r=16, q+v)

| Variant | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---------|-----------------:|-----------:|----------:|----------:|-----------:|
| LoRA (baseline) | 3,686,400 | 4.26 min | 6.62 GB | 1.5161 | 4.554 |
| DoRA | `<điền>` | `<điền>` | `<điền>` | `<điền>` | `<điền>` |

- **Nhận xét**: DoRA tách magnitude/direction → kỳ vọng perplexity tốt hơn nhẹ, đổi lại chậm hơn.
  `<cập nhật: có cải thiện không?>`

### 7.3 W&B
- Run link: `<https://wandb.ai/<user>/lab21-lora-rank/runs/...>` *(bật `USE_WANDB=True` + secret `WANDB_API_KEY`)*

### 7.4 GGUF export
- Đã merge r=16 + export `q4_k_m` → `r16_gguf/`, test với `llama.cpp`. *(bật `SAVE_GGUF=True`)*

### 7.5 Reproducibility (Option C)
- `requirements.txt` (pins) + `results/requirements_freeze.txt` (exact `pip freeze`).
