# Toàn tập lệnh chạy Pipeline (Zone-Aware-GNN)

Tất cả các lệnh dưới đây đều được chạy từ thư mục gốc của project (đảm bảo bạn đang đứng ở `/home/toon/Zone-Aware-GNN` và đã active môi trường ảo).

## 1. Chạy Unit Test
Lệnh này sẽ kiểm tra toàn bộ tính toàn vẹn của dữ liệu (data simulator), module chia tập, hàm tính metrics và code training xem có lỗi nào không trước khi chạy thật.
```bash
source venv/bin/activate
python -m pytest -q
```

## 2. Test luồng chạy nhanh (Smoke Test)
Lệnh này giúp kiểm tra xem mô hình có thể huấn luyện và sinh kết quả thành công với số lượng epoch nhỏ (vài epoch), để đảm bảo không bị lỗi RAM/VRAM hoặc bug pipeline.
```bash
python scripts/quick_test.py
```

## 3. Huấn luyện nhiều hạt giống (Multi-seed Training)
Lệnh này sẽ train các mô hình (nhóm đề xuất & baselines) trên nhiều seed ngẫu nhiên khác nhau. Việc chia tập sẽ dùng phương thức `chrono` (chia theo trục thời gian thực tế, tuân thủ không rò rỉ dữ liệu). Kết quả sẽ tự động lưu lại version mới nhất ở file CSV.
```bash
python scripts/run_multi_seed.py --models proposed baselines --split-modes chrono --seeds 42 43 --epochs 50
```
*(Bạn có thể thay đổi tham số `--epochs` hoặc truyền thêm nhiều `--seeds` tùy thời gian cho phép).*

## 4. Huấn luyện kịch bản chia cụm Non-IID
Lệnh này chạy mô phỏng các scenario Non-IID trên mạng giao thông, kiểm tra sức chịu đựng của các node/zone khi dữ liệu bị dịch chuyển phân phối (Domain Shift). Kết quả sẽ được ghi vào file `non_iid_eval.csv`.
```bash
python scripts/eval_non_iid.py --epochs 50
```

## 5. Xuất báo cáo tự động (Thống kê và LaTeX)
Sau khi có dữ liệu từ lệnh (3) và (4), module này sẽ tính toán: Mean±Std, Độ tin cậy CI95%, Effect size, Paired-test Holm correction, và tự động bóc tách (Breakdown) theo Horizon (t) và Scenario Alpha.
```bash
# Sinh tự động 2 file: 1 file Markdown và 1 file LaTeX ở thư mục data/results/
python scripts/generate_report.py
```

## 6. Xuất Báo Cáo PDF Bìa Chuẩn (Đồ án)
Lệnh này sẽ copy file LaTeX tự động sinh từ bước (5) vào bên trong template báo cáo cuối cùng và biên dịch ra thành file PDF hoàn chỉnh.
```bash
# 1. Copy file table kết quả mới nhất chèn vào thư mục báo cáo
cp $(ls -t data/results/report_chrono_*.tex | head -1) Source_Report/results.tex

# 2. Biên dịch file Báo cáo bằng pdflatex (Chạy 2 lần để mục lục được cập nhật số trang chuẩn xác)
cd Source_Report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
