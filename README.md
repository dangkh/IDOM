# IDOM — thí nghiệm bước 1: FMRec baseline và IDOM chưa có intent anchor

Code dựng trên bản chính thức của FMRec (IJCAI'25, https://github.com/FengLiu-1/FMRec).
Phần flow (`src/fmrec.py`) và phần dữ liệu (`src/utils.py`) giữ nguyên.

## Đã thay đổi gì so với FMRec gốc

| File | Thay đổi |
|---|---|
| `src/fmrec.py` | Bỏ import `timm`/`scipy` không dùng tới (scipy chỉ import khi gọi `rk45_sampler`). Logic giữ nguyên. |
| `src/utils.py` | `pin_memory` chỉ bật khi có CUDA. Ngoài ra không đổi gì. |
| `src/model.py` | `Att_FM_model` là FMRec gốc, loss giữ nguyên: `mse + 0.2·CE + 0.4·decoder`. `switch_Matrix` được vector hóa nhưng cho ra ma trận giống hệt. Thêm `IDOM_model`. |
| `src/trainer.py` | Cùng metric và cùng cách chọn model như FMRec (eval mỗi 20 epoch, patience 5, chọn theo điểm `fused`). Báo cáo thêm metric riêng của từng modality, thời gian train và inference, và lưu file JSON. Mặc định bỏ phần eval trên tập train (bật lại bằng `--eval_train`). |
| `src/main.py` | Thêm các tham số `--model`, `--modalities`, `--text_emb`, `--fuse_weights`, `--gpu_id`. FMRec gốc đặt cứng `CUDA_VISIBLE_DEVICES="7"`, phần này đã được bỏ. |

**Lưu ý:** FMRec gốc chỉ dùng ID, không có visual. Vì vậy "FMRec bỏ visual" chính là FMRec gốc.

## IDOM chưa có anchor (`--model idom`)

- Mỗi modality có một quỹ đạo flow riêng: quỹ đạo ID nằm trong không gian embedding ID, quỹ đạo text nằm trong không gian `text_proj(text_emb)`.
- **Mọi quỹ đạo dùng chung một vector field** (mạng `FM_xstart` của FMRec). Một modality embedding được cộng vào condition để mạng biết đang ở không gian nào. Các modality được ghép theo chiều batch (M·B), nên mỗi bước chỉ cần một lần forward.
- Mỗi modality được huấn luyện bằng đúng objective của FMRec trong không gian của nó, các loss được cộng lại.
- Khi suy luận, chạy Euler 30 bước cho từng modality, rồi tính `score = Σ_m w_m · log_softmax(x_m · E_mᵀ)`.
- **Chưa có ràng buộc nào giữa các modality.** Đây là đúng điểm để sau này thêm intent anchor.

## Cài đặt

```bat
conda create -n idom python=3.10 -y
conda activate idom
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Bước 0: Tạo text embedding cho Beauty (làm một lần)

Dữ liệu Beauty của FMRec đã đánh lại số item và không có text. Dữ liệu này trùng 100% với `Beauty.txt` của S3Rec,
nên script sẽ dựng lại mapping id → ASIN từ file raw và **kiểm tra mapping khớp tuyệt đối** trước khi ghi bất cứ thứ gì.

1. Tải hai file Amazon 2014 ở dòng "Beauty" tại https://jmcauley.ucsd.edu/data/amazon/links.html:
   `reviews_Beauty_5.json.gz` (5-core) và `meta_Beauty.json.gz`.
2. Chạy:

```bat
python scripts\prepare_beauty_text.py --reviews D:\data\reviews_Beauty_5.json.gz --meta D:\data\meta_Beauty.json.gz
```

Nếu thành công, script in ra `identical user sequences: 22363/22363` và tạo `datasets\data\amazon_beauty\text_emb_all-MiniLM-L6-v2.npy`.
Text của mỗi item gồm title, brand, category (bỏ category gốc "Beauty") và 64 từ đầu của description.
Có thể đổi encoder bằng `--encoder sentence-transformers/all-mpnet-base-v2`.

## Bước 1: Các lần chạy (từ thư mục `src`)

```bat
cd src
set EMB=..\datasets\data\amazon_beauty\text_emb_all-MiniLM-L6-v2.npy

:: (a) FMRec baseline, chỉ dùng ID
python main.py --model fmrec --gpu_id 0

:: (b) Kiểm tra sanity: IDOM với một modality ID, kết quả phải xấp xỉ (a)
python main.py --model idom --modalities id --gpu_id 0

:: (c) Một quỹ đạo text duy nhất
python main.py --model idom --modalities text --text_emb %EMB% --gpu_id 0

:: (d) IDOM chưa có anchor: hai quỹ đạo song song ID + text
python main.py --model idom --modalities id text --text_emb %EMB% --gpu_id 0
```

- Chạy thử nhanh: thêm `--epochs 41`. Với cấu hình này, eval chạy ở epoch 20 và 40.
- Để báo cáo kết quả: chạy mỗi cấu hình với 3 seed (`--random_seed 1997`, `2024`, `42`).
- Kết quả nằm ở `src\log\amazon_beauty\*.json`. Khóa `test.fused` là điểm chính. Với (d), file còn có `test.id` và `test.text` (điểm của từng quỹ đạo riêng lẻ).

## Cách đọc kết quả

- (b) ≈ (a): việc dùng chung vector field cộng modality embedding không làm hỏng FMRec.
- So (d).fused với (a): đây là lợi ích của việc sinh song song trong không gian text.
- So (d).id với (a): nếu (d).id cao hơn, nghĩa là quỹ đạo text giúp được cả vector field dùng chung.
- So (d).text với (c): cho biết huấn luyện chung có lợi hay có hại cho quỹ đạo text.
- Nếu (d).fused không vượt được max((a), (c)), thì cần intent anchor hoặc cách fusion tốt hơn. Thử thêm `--fuse_weights 1.0 0.5` trước.

## Bước tiếp theo (chưa làm)

- Thêm intent anchor trên các x̂₁(t) của từng modality.
- Thêm baseline "FMRec + text làm condition" (một quỹ đạo duy nhất, kiểu PrismRec) để bảo vệ đóng góp "sinh song song".
