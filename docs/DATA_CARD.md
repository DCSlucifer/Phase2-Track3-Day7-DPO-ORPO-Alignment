# Data Card — Preference Alignment Lab Dataset

**Học viên:** Võ Thành Danh - 2A202600503

## Thông tin chung

| Thuộc tính | Giá trị |
|---|---|
| **Tên dataset** | Sample Preference Pairs (Education Domain) |
| **Nguồn** | Tự tạo cho bài lab Preference Alignment |
| **Giấy phép** | Sử dụng cho mục đích giáo dục — không chứa nội dung bản quyền |
| **Kích thước** | 24 examples (24 prompts duy nhất, không trùng lặp) |
| **Ngôn ngữ** | Tiếng Anh (100%) |
| **Domain** | `education` (ML/DL concepts) — 100% |
| **Rubric** | `accuracy` — 100% |

## Schema

Mỗi dòng trong file `data/sample_preferences.jsonl` là một JSON object:

```json
{
  "prompt": "Explain the concept of self-attention in Transformers.",
  "chosen": "Self-attention allows the model to weigh the importance of different words...",
  "rejected": "Self-attention is a simpler version of RNNs that uses less memory...",
  "metadata": {"domain": "education", "rubric": "accuracy"}
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `prompt` | `string` | Câu hỏi/instruction về khái niệm ML |
| `chosen` | `string` | Câu trả lời chất lượng cao, chính xác về mặt khoa học |
| `rejected` | `string` | Câu trả lời hợp lý nhưng sai/thiếu chính xác |
| `metadata.domain` | `string` | Lĩnh vực — tất cả là `"education"` |
| `metadata.rubric` | `string` | Tiêu chí đánh giá — tất cả là `"accuracy"` |

## Labeling Rubric

Tiêu chí gán nhãn: **Accuracy** (Độ chính xác)
- **Chosen**: Giải thích đúng về mặt khoa học, chi tiết, đúng thuật ngữ
- **Rejected**: Có vẻ hợp lý nhưng chứa lỗi nhỏ, oversimplification, hoặc hallucination

**Ví dụ minh họa**:
- **Prompt**: *"How do GANs work?"*
- **Chosen** ✓: *"GANs consist of two networks, a generator and a discriminator, that compete against each other..."*
- **Rejected** ✗: *"GANs use a single network that generates data and then evaluates it..."* → **Sai**: GAN có 2 mạng, không phải 1

## Known Biases (Thiên lệch đã biết)

1. **Domain bias**: 100% examples thuộc lĩnh vực ML/DL education — không đại diện cho các domain khác (y tế, pháp luật, sáng tạo...)
2. **Length bias**: Chosen responses có xu hướng **dài hơn** rejected responses
   - Evidence: Length scorer đạt 100% accuracy → model có thể "gian lận" bằng cách chỉ ưu tiên response dài
3. **Language bias**: 100% tiếng Anh — không đánh giá được hiệu quả trên ngôn ngữ khác
4. **Rubric bias**: Chỉ có 1 tiêu chí (`accuracy`) — chưa test trên `helpfulness`, `harmlessness`, `creativity`

## Kiểm tra An toàn / PII

Kết quả quét tự động bằng `data.py`:

```
PII warnings: 0
```

| Loại PII | Pattern | Kết quả |
|---|---|---|
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | **0 phát hiện** |
| Số điện thoại | `\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` | **0 phát hiện** |
| SSN | `\b\d{3}-\d{2}-\d{4}\b` | **0 phát hiện** |

**Nội dung**: Tất cả là kiến thức giáo dục ML — không chứa nội dung có hại, độc hại, hoặc nhạy cảm.

**Near-duplicate**: Kiểm tra bằng SequenceMatcher (ngưỡng >95% similarity) — **0 cặp trùng lặp**.

## Phương pháp Split Train/Val

| Thuộc tính | Giá trị |
|---|---|
| Tỷ lệ | 80% train / 20% val |
| Seed | 42 (deterministic, có thể tái tạo) |
| Train | 20 examples, 20 unique prompts |
| Val | 4 examples, 4 unique prompts |
| Overlap | **0** — không leakage |

**Phương pháp**: Nhóm examples theo prompt text (case-insensitive, chuẩn hóa khoảng trắng). Shuffle theo seed, split ở mức nhóm prompt → đảm bảo **cùng một prompt không xuất hiện ở cả train và val**.

**Evidence — Code thực tế** (file `src/preference_lab/data.py`):
```python
def split_by_prompt(examples, validation_ratio=0.2, seed=42):
    """Split với nhóm theo prompt để tránh data leakage."""
    prompt_groups: dict[str, list] = {}
    for ex in examples:
        key = ex.prompt.strip().lower()
        prompt_groups.setdefault(key, []).append(ex)
    keys = sorted(prompt_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_size = max(1, int(len(keys) * validation_ratio))
    val_keys = set(keys[:val_size])
    # Split theo nhóm prompt — không leakage
    train = [ex for k, exs in prompt_groups.items() if k not in val_keys for ex in exs]
    val = [ex for k, exs in prompt_groups.items() if k in val_keys for ex in exs]
    return train, val
```
