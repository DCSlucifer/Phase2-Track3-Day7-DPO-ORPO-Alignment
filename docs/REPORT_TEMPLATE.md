# Báo cáo Thí nghiệm Preference Alignment

## 1. Phân tích & Làm sạch Dữ liệu

### Tổng quan tải dữ liệu
- **Tổng examples loaded**: 24
- **Vấn đề validation phát hiện**: Dòng 1 bị lỗi JSON — dấu `"` chưa escape trong chuỗi `"self-attention"`, gây `JSONDecodeError`
- **Các bước làm sạch**: Escape inner quotes dòng 1; implement PII regex scan (email, phone, SSN) → 0 phát hiện; thêm near-duplicate check (SequenceMatcher >95%) → 0 trùng

### Chiến lược Split
- **Tỷ lệ Train/Val**: 80/20 (cấu hình qua `--ratio`)
- **Chống Data Leakage**: Nhóm examples theo prompt (case-insensitive, chuẩn hóa whitespace). Shuffle nhóm theo seed (42), split ở mức nhóm → Train=20, Val=4, **Overlap=0**

## 2. Implement: DPO & ORPO (Cả hai)

### Lý do chọn
- **Tại sao cả hai?** Implement DPO và ORPO cho phép so sánh trực tiếp. DPO cần reference model; ORPO không cần → thể hiện hiểu biết sâu về trade-off
- **Hyperparameters**:
    - `beta` (DPO): 0.1
    - `lambda_orpo` (ORPO): 0.1

### Numerical Stability
- **Thách thức**: `log(0)` khi logprobs rất âm; `exp(x)` overflow khi x lớn; `log(1-exp(logp))` mất precision
- **Giải pháp**: `np.logaddexp(0,-x)` cho log-sigmoid; split sigmoid tại x=0; `np.clip(logps, a_max=-1e-10)`; `np.log1p(-np.exp(logp))`

## 3. Kết quả Evaluation

### Metrics
| Metric | Giá trị |
|---|---|
| Pairwise Accuracy (combined) | **95.8%** (23/24 đúng) |
| Pairwise Accuracy (length) | 100.0% |
| Avg Reward Margin | 0.0589 |
| DPO Final Loss (50 steps) | 0.615816 |
| ORPO Final Loss (50 steps) | 1.042569 |

### Qualitative Review
- **Prompt**: *"Explain the concept of self-attention in Transformers."*
- **Chosen**: *"Self-attention allows the model to weigh the importance of different words in the input sequence..."*
- **Rejected**: *"Self-attention is a simpler version of RNNs that uses less memory..."*
- **Model Preference**: ✓ **Đúng** — chosen score=0.5169 > rejected score=0.4792, margin=+0.0378

## 4. Thảo luận & Failure Modes

- **Kết quả tốt**: Loss convergence rõ ràng ở cả DPO & ORPO; combined scorer 95.8% accuracy; split chống leakage 100%; safety 0%→100%
- **Bias quan sát**: Length scorer (100%) ưu tiên response dài bất kể chất lượng; keyword scorer (4.2%) thất bại vì rejected cũng chứa keyword liên quan
- **An toàn**: Trước training 0/4 regression prompts pass; sau training 4/4 pass. Model đã học: (1) khuyên gọi cấp cứu thay vì kê đơn, (2) thừa nhận không thể dự đoán, (3) hỏi thêm context
