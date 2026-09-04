# docs/ — Nhật ký kỹ thuật

Mỗi file ghi lại: **lỗi gì / vì sao phải sửa / vì sao sửa theo cách đó**.
Đọc theo thứ tự số nếu mới vào.

| File | Nội dung | Mức |
|---|---|---|
| [`00_dataset_provenance.md`](00_dataset_provenance.md) | `graph_dataset.pt` là dữ liệu **synthetic**, không phải TomTom — bằng chứng tái tạo bit-exact, hệ quả với paper, việc nhóm phải quyết | 🔴 Chặn |
| [`01_bugfix_train_lambda_cos.md`](01_bugfix_train_lambda_cos.md) | `train.py` crash ngay khi chạy (`NameError` + `TypeError` quanh `lambda_cos`) | 🔴 Chặn |
| [`02_bugfix_time_label_map.md`](02_bugfix_time_label_map.md) | `TIME_LABEL_MAP` nuốt mất nhãn giờ cao điểm — `time_labels` chỉ còn `{0, 3}` | 🟠 Cao |
| [`03_benchmark_partition_design.md`](03_benchmark_partition_design.md) | Thiết kế bộ sinh phân vùng Non-IID: Dirichlet chia **độ phủ thời gian theo node**, spec chống rò rỉ, contract cho Người 4 | 🟠 Cao |

---

## Ba việc cần quyết sớm nhất

1. **Tên dataset trong paper** — `docs/00` §6. Chặn Data Card của Người 4.
2. **Thứ tự fix** — `docs/02` §6: sửa `TIME_LABEL_MAP` **trước** khi Người 4 chạy experiment,
   nếu không họ phải chạy lại toàn bộ.
3. **Đừng debug `zone_weight` / `zone_adj`** — `docs/00` §4.2 giải thích tại sao chúng
   không hỏng, chỉ sai giả định về dữ liệu.
