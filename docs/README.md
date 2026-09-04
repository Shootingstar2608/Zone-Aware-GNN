# docs/

Nhật ký kỹ thuật của dự án. Mỗi file ghi: **lỗi gì / vì sao phải sửa / vì sao sửa theo cách đó**.

---

## ⚠️ Đọc trước — ảnh hưởng cả nhóm

### [`khoa/00_dataset_provenance.md`](khoa/00_dataset_provenance.md) 🔴

`graph_dataset.pt` là **dữ liệu synthetic**, không phải TomTom. Đã xác minh bằng tái tạo
bit-exact từ `scripts/dev/generate_synthetic_traffic.py`.

Ba việc cả nhóm phải quyết **trước khi Người 4 viết Data Card**:

1. **Tên dataset trong paper** — không được ghi "TomTom Routing API"
2. **Bảng kết quả cũ** giữ hay bỏ, và giữ với vai trò gì
3. **Có thu dữ liệu TomTom thật không** — nếu có thì phải bắt đầu ngay, chạy mất nhiều ngày

Và một việc **không** phải làm: **đừng debug `zone_weight` / `zone_adj`** — §4.2 giải thích
tại sao chúng không hỏng, chỉ sai giả định về dữ liệu. Đừng ai đốt một tuần vào đó.

---

## Theo người

| Thư mục | Người | Nội dung |
|---|---|---|
| [`khoa/`](khoa/) | Khoa (Người 2) | Provenance dataset, 2 bugfix chặn, thiết kế bộ sinh phân vùng Non-IID |

---

## Quy ước

- Đặt tên `NN_ten_ngan.md`, số tăng dần theo thời gian trong từng thư mục
- Mỗi doc nêu rõ **mức độ** (🔴 chặn / 🟠 cao / 🟡 vừa) và **ảnh hưởng tới ai**
- Doc mô tả một fix làm **hỏng khả năng so sánh với kết quả cũ** thì phải nói thẳng ở đầu file
