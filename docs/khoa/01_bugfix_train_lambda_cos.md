# 01 — Bugfix: `scripts/train.py` crash ngay khi chạy (`lambda_cos`)

**Mức độ:** 🔴 Chặn — `train.py` hiện **không chạy được ở bất kỳ chế độ nào**
**Ảnh hưởng tới:** Người 4 (`scripts/eval_non_iid.py` gọi vào pipeline này), `scripts/run_multistep.py`
**Chi phí sửa:** 1 dòng

---

## 1. Triệu chứng

Chạy bất kỳ lệnh nào trong README đều nổ:

```bash
python scripts/train.py --ablation
python scripts/train.py --variant zone_full
```

---

## 2. Nguyên nhân chính xác

Có **hai lỗi chồng nhau**, cùng xoay quanh biến `lambda_cos`.

### Lỗi A — `NameError` trong thân hàm (dòng 372 & 377)

```python
def run_experiment(variant_name, meta, dataset_dict, ablation_cfg):   # <- khong co lambda_cos
    set_seed(42)
    use_zone_emb, use_zone_weight, use_zone_adj = ablation_cfg

    effective_lambda_cos = lambda_cos if variant_name in ZONE_AWARE_VARIANTS else 0.0
    #                      ^^^^^^^^^^ chua duoc dinh nghia o bat ky scope nao
```

Module có hằng `LAMBDA_COS` (viết **hoa**), nhưng `lambda_cos` (viết **thường**) thì không
tồn tại — không phải tham số, không phải biến global, không phải builtin.

⇒ `NameError: name 'lambda_cos' is not defined`

### Lỗi B — `TypeError` ở chỗ gọi (dòng 559)

```python
result = run_experiment(vname, meta, dataset, vcfg, lambda_cos=args.lambda_cos)
#                                                   ^^^^^^^^^^ signature khong nhan kwarg nay
```

⇒ `TypeError: run_experiment() got an unexpected keyword argument 'lambda_cos'`

Lỗi B nổ **trước** lỗi A, nên khi chạy `main()` sẽ chỉ thấy `TypeError`. Nhưng sửa mỗi B là
chưa đủ — A vẫn còn nằm đó, và `run_multistep.py` (gọi 4 tham số, không kwarg) sẽ dính A.

### Vì sao lọt lưới

Đây là **vết khâu của một lần merge**. Tính năng cosine regularization (`--lambda-cos`) do
Bảo thêm vào; nhánh còn lại sửa `run_experiment` để lưu checkpoint theo `T_out`. Khi gộp
`origin/ton` + `origin/khoa` + PR #4, phần thân hàm lấy từ nhánh có tham số, còn dòng
`def` lấy từ nhánh không có. Không ai chạy lại `train.py` trực tiếp sau merge vì mọi người
đều chạy qua `run_multistep.py` hoặc `run_multi_seed.py`.

> `run_multi_seed.py` **không** import `run_experiment` — nó có `run_once()` riêng.
> Đó là lý do 180 runs multi-seed vẫn chạy được trong khi `train.py` đã hỏng.

---

## 3. Cách sửa

Thêm tham số vào signature, mặc định là hằng có sẵn:

```python
# scripts/train.py — dong 372
def run_experiment(variant_name, meta, dataset_dict, ablation_cfg, lambda_cos=LAMBDA_COS):
```

Chỉ vậy. Không đụng gì thêm.

---

## 4. Vì sao sửa theo cách này mà không phải cách khác

| Phương án | Đánh giá |
|---|---|
| **Thêm param có default `=LAMBDA_COS`** ✅ | Sửa cả A và B bằng 1 dòng. Giữ nguyên hành vi mặc định ⇒ kết quả cũ vẫn tái lập. `run_multistep.py` gọi 4 tham số vẫn chạy nhờ default. |
| Đổi `lambda_cos` → `LAMBDA_COS` trong thân hàm | Sửa được A nhưng **không** sửa B. Và làm chết cờ `--lambda-cos` — mất luôn tính năng Bảo viết. |
| Bỏ `lambda_cos=` ở dòng 559 | Sửa được B nhưng **không** sửa A. Cũng làm cờ CLI thành vô nghĩa. |
| Đặt `lambda_cos` thành biến global | Chạy được, nhưng biến toàn cục mutable giữa các experiment ⇒ chạy nhiều variant liên tiếp có thể rò trạng thái. Tệ hơn hẳn. |

Nguyên tắc chọn: **sửa tối thiểu, không đổi hành vi mặc định, không làm mất tính năng của người khác.**
Giá trị mặc định `LAMBDA_COS = 0.1` giữ nguyên nên mọi số liệu đã có vẫn so sánh được với số liệu mới.

---

## 5. Kiểm chứng sau khi sửa

```bash
# 1. Chay duoc, khong crash
python scripts/train.py --variant gcn_gru

# 2. Co gcn_gru la baseline -> phai in "cosine_reg=OFF"
#    (vi gcn_gru khong nam trong ZONE_AWARE_VARIANTS)

# 3. Variant zone -> phai in "cosine_reg=ON (lambda=0.1)"
python scripts/train.py --variant zone_concat

# 4. Co CLI van co tac dung
python scripts/train.py --variant zone_concat --lambda-cos 0.0
#    -> phai in "cosine_reg=OFF"
```

Nếu bước 4 vẫn in `ON` thì tham số chưa được truyền xuống — kiểm tra lại dòng 559.

---

## 6. Việc kèm theo (không bắt buộc, nhưng nên)

`scripts/train.py` dòng 25 có `from torchgen import model` — import thừa, không dùng ở đâu,
và `torchgen` là module nội bộ của quá trình build PyTorch, không đảm bảo tồn tại ở mọi bản cài.
Xoá đi để tránh `ImportError` trên máy khác (ví dụ máy của Người 4).
