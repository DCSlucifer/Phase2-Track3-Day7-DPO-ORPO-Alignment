# Data Card


- **Tên dataset**: Sample Preference Pairs (Education Domain) — 24 cặp preference về ML/DL
- **Nguồn**: Tự tạo cho bài lab Preference Alignment, file `data/sample_preferences.jsonl`
- **Giấy phép**: Sử dụng cho mục đích giáo dục — không chứa nội dung bản quyền
- **Schema**: Mỗi entry gồm `prompt` (string), `chosen` (string), `rejected` (string), `metadata` (object: `domain`, `rubric`)
- **Labeling rubric**: **Accuracy** — chosen response phải chính xác về mặt khoa học; rejected response có vẻ hợp lý nhưng chứa lỗi
- **Known biases**: (1) 100% domain education, (2) chosen responses dài hơn rejected, (3) 100% tiếng Anh, (4) chỉ 1 rubric "accuracy"
- **Kiểm tra An toàn/PII**: Quét tự động 3 loại PII (email, phone, SSN) bằng regex → **0 phát hiện**. Near-duplicate check (>95% similarity) → **0 trùng**
- **Phương pháp split Train/Val**: Nhóm theo prompt text (case-insensitive) rồi split ở mức nhóm. Tỷ lệ 80/20, seed=42. Kết quả: Train=20, Val=4, Overlap=**0** (không leakage)
