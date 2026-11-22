# Bảng Kiểm tra Đối chiếu Tuyên bố Khoa học & Bằng chứng Thực nghiệm (Claim-Evidence Checklist)

Tài liệu này đóng vai trò là danh mục kiểm tra liêm chính học thuật bắt buộc (Mandatory Scientific Integrity Checklist) dành cho nhóm nghiên cứu trước khi phát hành Báo cáo Môn học (`Source_Report/`) hoặc Bài báo Khoa học (`paper/`). 

Tất cả các khẳng định (claims) xuất hiện trong văn bản báo cáo bắt buộc phải được đối soát với bằng chứng thực nghiệm cụ thể tại bảng dưới đây:

---

## 1. Bảng Đối chiếu Tuyên bố Khoa học (Claim-Evidence Matrix)

| STT | Khẳng định Khoa học (Research Claim) | Trạng thái Cho phép | Bằng chứng Thực nghiệm & Mã nguồn Đối chứng | Hướng dẫn Viết trong Báo cáo / Paper |
|:---:|---|:---:|---|---|
| **1** | **HCM-Sim v2 là dữ liệu lưu lượng thực địa của TP.HCM** | ❌ **TUYỆT ĐỐI CẤM** | - Thuật toán sinh tại `scripts/dev/generate_synthetic_traffic.py`<br>- Bằng chứng Data Card tại `docs/data_card_hcm_sim.md` | **BẮT BUỘC** gọi là: "Dữ liệu mô phỏng tổng hợp (Synthetic Traffic Benchmark) kết hợp đồ thị không gian OSRM thực tế và hàm tuần hoàn nhật kỳ". |
| **2** | **Nghiên cứu đã hoàn thiện hệ thống Học liên kết (Federated Learning)** | ❌ **TUYỆT ĐỐI CẤM** | - Mã nguồn mới chỉ dừng ở mức phân vùng Non-IID tại `benchmark/partition_gen.py`<br>- Chưa có bộ điều phối tổng hợp trọng số `FedAvg` hoặc `FedProx` | **BẮT BUỘC** gọi là: "Thử nghiệm sơ bộ khả năng chống chịu trước phân phối lệch số lượng (Quantity-Skew Non-IID Robustness Pilot)". |
| **3** | **Seasonal Naive thắng hợp lệ và không có hiện tượng rò rỉ dữ liệu (Data Leakage)** | ✅ **ĐƯỢC PHÉP KHẲNG ĐỊNH** | - 04 unit tests nhân quả tại `tests/test_temporal_benchmark.py` pass 100%<br>- MAE đạt **0.0663 ± 0.0000** qua 5 hạt giống tại `data/results/hcm_sim_v2/clean_5seed_statistics_summary.csv` | Báo cáo minh bạch tính ưu thế tuyệt đối của Seasonal Naive; lý giải do tính tất định mạnh mẽ của chu kỳ tuần hoàn ngày đêm trong hàm sinh dữ liệu. |
| **4** | **Zone-Aware AH-GNN vượt trội có ý nghĩa thống kê trong nhóm Deep Learning** | ✅ **ĐƯỢC PHÉP KHẲNG ĐỊNH** | - Đạt MAE **0.1108 ± 0.0026** (thấp nhất trong 5 mô hình học sâu)<br>- Kiểm định paired t-test hiệu chỉnh Holm đạt **$p < 0.001$** so với GCN-GRU (MAE 0.1274)<br>- Giảm 3.0% MAE so với khi bỏ nhãn vùng (0.1142) | Khẳng định nhãn ngữ nghĩa vùng đô thị (Zone Semantics) mang lại đóng góp khoa học có ý nghĩa thống kê giúp nâng cao năng lực dự báo không-thời gian. |
| **5** | **Dữ liệu thực địa TomTom v1-pilot đã sẵn sàng cho huấn luyện dự báo dài hạn** | ❌ **TUYỆT ĐỐI CẤM** | - Báo cáo chất lượng `tailieu/hcm_real_quality_report.md` chỉ ra tỷ lệ khuyết thiếu **42.29%** (4,479 dòng)<br>- Chuỗi liên tục dài nhất đạt 35 giờ $\rightarrow$ **0 cửa sổ trượt liên tục 36 bước ($T_{in}=12, T_{out}=24$)** | Báo cáo rõ đây là bộ dữ liệu thực nghiệm sơ bộ (v1-pilot) phục vụ kiểm định collector, chưa áp dụng trực tiếp cho benchmark dài hạn nếu không có giải thuật nội suy. |
| **6** | **Quy trình thực nghiệm bảo đảm tính tái lập hoàn toàn từ clone sạch (Bit-for-bit Reproducibility)** | ✅ **ĐƯỢC PHÉP KHẲNG ĐỊNH** | - Toàn bộ 31 phân vùng Non-IID khớp 100% mã băm `mask_hash` khi kiểm tra bằng `scripts/verify_partitions.py`<br>- Kịch bản kiểm định độc lập `scripts/self_test_reproduction.py` chạy thành công từ clone sạch | Khẳng định mã nguồn và dữ liệu đáp ứng tiêu chuẩn tái lập khoa học cao nhất. |

---

## 2. Tiêu chuẩn Thẩm định Trước khi Nộp Báo cáo (Pre-submission Checklist)

- [x] Không có bất kỳ đoạn văn nào dùng từ "dữ liệu thực", "quan sát thực tế" khi nhắc đến HCM-Sim v2.
- [x] Không dùng cụm từ "thuật toán Federated Learning hoàn chỉnh", "kết quả huấn luyện FedAvg/FedProx" khi chưa có code máy chủ tổng hợp.
- [x] Bảng kết quả chính thức trong báo cáo (`Source_Report/results.tex`) bắt buộc phải có `seasonal_naive` ở vị trí hàng đầu tiên với số liệu in đậm.
- [x] Phải có phần phân tích khoa học về Hạn chế nghiên cứu (Limitations): tính tuần hoàn nhân tạo của HCM-Sim, tỷ lệ thiếu dữ liệu của TomTom v1-pilot, và giới hạn toán học của cỡ mẫu kiểm định $n=5$.
- [x] Bộ kiểm thử nhân quả và tính tái lập đạt 100% trạng thái Pass (`pytest -q`).

