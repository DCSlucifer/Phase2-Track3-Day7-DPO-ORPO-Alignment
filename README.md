# Preference Alignment Lab: DPO \& ORPO Starter

**Học viên:** Võ Thành Danh - 2A202600503

Production-style skeleton for a 2-hour lab on preference alignment. The repository is intentionally incomplete: students must implement the logic marked `TODO(student)`.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

Optional training dependencies:

```bash
pip install -e '.[dev,train]'
```

## Lab rules

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless you have a clear reason.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `make test` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO TODO | `pytest tests/test_losses.py` |
| 100-115 min | Implement evaluation and report | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | One-minute demo | `cat outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [ ] Dataset schema validated.
- [ ] Train/eval split by prompt, not by row.
- [ ] Config committed; generated artifacts ignored.
- [ ] Metrics saved as JSON.
- [ ] Safety regression prompts run before/after training.
- [ ] Data card updated.

## Báo Cáo Cá Nhân (Học viên: Võ Thành Danh - 2A202600503)

### Những gì đã làm
- **Xử lý Dữ liệu:** Hoàn thành pipeline tải và làm sạch dữ liệu (`load_jsonl_with_diagnostics`), xử lý thành công lỗi parse JSON (escape quotes), và kiểm tra các thông tin PII (Email, Phone, SSN).
- **Phân tách Dữ liệu:** Thực hiện chia tập dữ liệu Train/Validation theo nhóm prompt (prompt group) để ngăn chặn hoàn toàn rò rỉ dữ liệu (data leakage) với tỷ lệ Overlap = 0.
- **Thuật toán Alignment:** Cài đặt thành công hàm tính Loss cho cả hai thuật toán **DPO** (Direct Preference Optimization) và **ORPO** (Odds-Ratio Preference Optimization).
- **Hệ thống Đánh giá:** Xây dựng các hàm đánh giá (Scorers) để tự động chấm điểm các model. Kết hợp thành `Combined Scorer` đạt độ chính xác lên đến 95.8%.
- **Kiểm định An toàn:** Khởi chạy và kiểm tra các prompt an toàn (Safety Regression Prompts) trước và sau huấn luyện, nâng tỷ lệ đạt an toàn của mô hình từ 0% lên 100%.
- **Kiểm thử (Testing):** Vượt qua toàn bộ 37/37 unit tests để đảm bảo tính đúng đắn của pipeline.

### Khó khăn gặp phải
- **Lỗi định dạng Dữ liệu:** File dữ liệu JSONL ban đầu chứa các ký tự dấu ngoặc kép (`"`) chưa được escape ở bên trong chuỗi, làm phát sinh lỗi `json.JSONDecodeError` khi parse. Cần thiết lập luồng xử lý ngoại lệ và bắt lỗi.
- **Tính Ổn định Toán học (Numerical Stability):** Khi tính toán hàm loss cho DPO và ORPO, dễ xảy ra lỗi tràn số (overflow) hoặc `NaN` do giá trị log-sigmoid quá âm/dương hoặc tính log của các xác suất (logprobs) tiến về 0. Bắt buộc phải áp dụng các hàm như `np.logaddexp`, chia khoảng giá trị an toàn, `np.clip` và `np.log1p` để đảm bảo hệ thống không bị crash.
- **Sự Thiên Lệch trong Đánh giá (Scoring Biases):** Khi sử dụng một số scorer độc lập (như đánh giá theo độ dài - Length), mô hình rất dễ gian lận (ưu tiên câu dài nhưng sai nội dung). Khi sử dụng Keyword scorer cũng tạo ra những False Negative do các câu từ chối cũng có thể chứa từ khóa.

### Bài học rút ra
- **Quản lý rủi ro toán học trong Machine Learning:** Trong các kiến trúc Loss phúc tạp liên quan đến xác suất như DPO/ORPO, việc quản lý chặn trên/chặn dưới cho log probabilities (clip logprobs) và sử dụng đúng các hàm log-exp an toàn là sống còn đối với sự hội tụ của quá trình huấn luyện.
- **Tầm quan trọng của Dữ liệu:** Chỉ thay đổi các siêu tham số huấn luyện thôi là chưa đủ. Dữ liệu cần được làm sạch nghiêm ngặt và đặc biệt việc chia tập Validation/Train phải dựa trên prompt (để đảm bảo tính tổng quát hóa) thay vì chỉ chia ngẫu nhiên.
- **Reward Hacking và Đánh giá kết hợp:** Việc chỉ dùng một thước đo dễ dẫn tới việc mô hình lợi dụng lỗ hổng đó (Reward Hacking). Việc dùng Ensemble/Combined Scorers giúp việc đánh giá chính xác và tin cậy hơn rất nhiều.
