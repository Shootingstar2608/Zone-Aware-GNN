# Kết quả đa seed — cảnh báo quan trọng cho narrative của paper

**Thiết lập:** 9 model × 10 seed (42–51) × 2 giao thức chia dữ liệu = 180 lần train.
Sinh bởi `scripts/run_multi_seed.py`, phân tích bởi `scripts/stat_analysis.py`.
Số liệu thô: `multiseed_runs.csv`. Báo cáo đầy đủ: `multiseed_report.txt`.

> Chạy trên CPU 2 nhân mất ~36 phút cho 180 lần train. Trên máy có GPU sẽ nhanh hơn nhiều.

---

## 1. Bảng chính (split ngẫu nhiên, seed chia cố định = 42 — giống train.py gốc)

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| LSTM | 0.2249 ± 0.0105 | 0.3300 ± 0.0275 | 15.72 ± 0.57 |
| STGCN | 0.0948 ± 0.0038 | 0.1681 ± 0.0060 | 6.63 ± 0.30 |
| GCN-GRU | 0.0936 ± 0.0017 | 0.1651 ± 0.0037 | 6.60 ± 0.17 |
| AH-GNN (no zone) | 0.2032 ± 0.0323 | 0.3320 ± 0.0965 | 15.36 ± 2.44 |
| **+ zone concat** | **0.0784 ± 0.0056** | **0.1340 ± 0.0059** | **5.83 ± 0.39** |
| + zone weight | 0.0932 ± 0.0120 | 0.1526 ± 0.0158 | 6.89 ± 0.81 |
| zone_full (Proposed cũ) | 0.0918 ± 0.0103 | 0.1467 ± 0.0120 | 6.91 ± 0.74 |
| + sinusoidal time | 0.0801 ± 0.0038 \*\*\* | 0.1364 ± 0.0051 \*\*\* | 5.99 ± 0.26 \*\*\* |
| + time-conditioned | 0.0824 ± 0.0077 \*\* | 0.1399 ± 0.0113 \*\*\* | 6.17 ± 0.54 \* |

\* p<0.05, \*\* p<0.01, \*\*\* p<0.001 so với GCN-GRU, t-test đã hiệu chỉnh Holm.

---

## 2. Ba phát hiện làm thay đổi narrative

### 2.1. Bảng single-run hiện tại là NHIỄU, không phải kết quả

`all_results.csv` (1 seed) xếp hạng: `zone_full_tc` (0.0773) tốt nhất, `zone_concat` (0.0860) đứng thứ 3.
Với 10 seed thì thứ tự **đảo ngược**: `zone_concat` (0.0784) tốt nhất, `zone_full_tc` (0.0824) đứng thứ 3.

Độ lệch chuẩn của các zone model là 0.004–0.012 MAE, trong khi khoảng cách giữa chúng chỉ 0.002–0.005.
Nói cách khác: **nhiễu do seed lớn hơn hiệu ứng đang cố đo**. Mọi kết luận rút ra từ 1 lần chạy đều không đứng vững.

### 2.2. Ablation study đang chứng minh điều NGƯỢC LẠI với giả thuyết

Kiểm định `zone_concat` làm mốc (paired t-test, Holm, n=10):

| So sánh | Chênh lệch MAE | Thắng | p (Holm) | Kết luận |
|---|---|---|---|---|
| zone_full vs zone_concat | **−17.0%** (tệ hơn) | 2/10 | 0.030 \* | zone_full **tệ hơn có ý nghĩa** |
| zone_weight vs zone_concat | **−18.9%** (tệ hơn) | 1/10 | 0.030 \* | tệ hơn có ý nghĩa |
| zone_full_tc vs zone_concat | −5.0% | 4/10 | 0.57 | không khác biệt |
| zone_full_sinc vs zone_concat | −2.1% | 5/10 | 0.57 | không khác biệt |

Đọc thẳng: **hai thành phần `zone_weight` và `zone_adj` không những vô ích mà còn làm hỏng model.**
Thứ duy nhất tạo ra cải thiện là bước đơn giản nhất — nối zone embedding vào feature.
Kết quả này lặp lại y hệt ở cả hai giao thức chia dữ liệu, nên không phải do một split xui.

Đây là vấn đề nghiêm trọng vì đóng góp khoa học #1 của paper là "zone-modulated convolution +
zone-aware adjacency". Có 3 hướng xử lý, cần chọn trước khi viết Section 4:

1. **Debug** — nhiều khả năng có bug thật (ví dụ `alpha` trộn adjacency quá lớn, hoặc
   zone-modulated weight làm số tham số nổ lên 358k trên 17 node → overfit). 17 node mà 358k tham số
   là tỉ lệ rất đáng ngờ; GCN-GRU chỉ 32k tham số mà gần bằng.
2. **Đổi narrative** — bài báo trở thành "zone semantics giúp ích, nhưng cách đưa vào đơn giản nhất
   lại tốt nhất; các cơ chế phức tạp hơn overfit trên đồ thị nhỏ". Đây là kết quả negative nhưng
   trung thực và vẫn đăng được.
3. **Giữ nguyên và báo cáo single-run** — không nên. Reviewer chỉ cần yêu cầu error bar là sập.

### 2.3. `zone_full_sinc` (Bảo) ổn định hơn `zone_full_tc` (Tân)

`zone_full_sinc` thắng GCN-GRU **10/10 seed** trên cả 3 metric, p < 0.001 sau Holm ở mode cố định.
`zone_full_tc` thắng 8–9/10 và ở mode `random_paired` thì MAE và MAPE **không đạt** ý nghĩa thống kê.
Nếu chỉ chọn một variant để làm "đóng góp #2", sinusoidal là lựa chọn an toàn hơn.

---

## 3. Vấn đề phương pháp luận CHƯA xử lý: rò rỉ thời gian

`train.py` dùng `random_split` trên chuỗi thời gian. Dataset là 658 cửa sổ trượt (T_in=12, T_out=3)
từ ~7 ngày dữ liệu. Chia ngẫu nhiên nghĩa là một cửa sổ test có thể bắt đầu **chỉ 1 bước** sau một
cửa sổ train → 11/12 bước đầu vào trùng nhau. Model gần như đã "nhìn thấy" đáp án.

Đây là lý do MAE ≈ 0.08 và MAPE ≈ 6% trông đẹp bất thường. DCRNN, STGCN, Graph WaveNet — tức đúng
những paper bạn định cite — đều chia **theo thứ tự thời gian** 70/10/20. Reviewer mảng traffic
forecasting sẽ bắt lỗi này ngay ở vòng đầu.

Script đã hỗ trợ sẵn:

```bash
python scripts/run_multi_seed.py --models all --seeds 42 43 44 45 46 47 48 49 50 51 \
    --split-modes chrono --resume
python scripts/stat_analysis.py
```

Dự đoán: MAE sẽ tăng đáng kể ở tất cả model. Điều quan trọng không phải con số tuyệt đối mà là
**thứ hạng có giữ nguyên không**. Nếu zone models vẫn thắng baselines dưới chia theo thời gian →
kết quả vững chắc. Nếu không → phải viết lại toàn bộ phần thực nghiệm.

---

## 4. Ghi chú thống kê

- **Paired vs Welch:** ở `random_paired`, seed s cho model A và model B dùng chung tập test →
  paired t-test hợp lệ và mạnh hơn. Ở `random_fixed`, tập test giống hệt nhau nên seed chỉ đổi
  khởi tạo → ghép cặp vô nghĩa, phải dùng Welch. Cột `recommended_test` trong
  `multiseed_significance.csv` đánh dấu cái nên trích dẫn.
- **Tại sao 10 seed chứ không phải 5:** Wilcoxon signed-rank chính xác có p hai phía nhỏ nhất
  = 2/2^n. Với n=5 thì bằng 0.0625 — **không bao giờ** đạt p<0.05 dù thắng tuyệt đối 5/5.
  Với n=10, sàn là 0.002 → kiểm định phi tham số dùng được. Vì vậy đã chạy 10 seed.
- **Đa so sánh:** 2 model × 3 metric = 6 kiểm định mỗi họ. Không hiệu chỉnh thì xác suất có ít nhất
  một dương tính giả ≈ 26%. Đã áp dụng Holm–Bonferroni trong từng (split_mode × loại kiểm định).
- **Effect size:** Cohen's d_z và Hedges' g đều nằm trong `multiseed_significance.csv`. Các so sánh
  có ý nghĩa đều có |g| > 2 (rất lớn), nên p nhỏ không phải do phương sai nhỏ giả tạo.
