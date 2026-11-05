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

## Chạy trên GPU server (PBS)

Các job dùng cùng cấu hình với `dragon_diag.pbs`: queue `small_gpuq`, `select=1:ncpus=4:ngpus=1:mem=32gb`,
`~/miniconda3` và env `Astar`. Đổi env bằng `-v CONDA_ENV=...`. Lệnh chạy từ thư mục gốc của repo trên server.

```bash
# 0) Tạo text embedding (một lần): chạy thẳng trên login node bằng CPU, không cần GPU và không qua queue, mất vài phút.
#    Bước này dùng env CPU riêng tên idomtext, để không đụng tới torch 2.5.1 của env Astar. Tạo env một lần:
conda create -n idomtext python=3.11 -y && conda activate idomtext
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install sentence-transformers numpy
conda deactivate
#    Thư mục truyền vào phải chứa reviews_Beauty_5.json.gz và meta_Beauty.json.gz
bash scripts/prepare_text.sh ~/ihpc_data/hkieu/amazon2014

# 1) Chạy thử nhanh: 4 job (mỗi cấu hình một job), 1 seed, 41 epoch
EPOCHS=41 SEEDS=1997 bash scripts/submit_all.sh

# 2) Chạy đầy đủ: vẫn chỉ 4 job. Mỗi job chạy 3 seed song song trên cùng một GPU.
#    Seed nào đã có kết quả sẽ tự được bỏ qua; output của từng seed nằm ở src/log/amazon_beauty/<run>.out
bash scripts/submit_all.sh

# Gửi riêng một cấu hình
qsub -v CONFIG=idom_idtext scripts/idom_run.pbs
qsub -v CONFIG=idom_idtext,TAG=w05,EXTRA_ARGS="--fuse_weights 1.0 0.5" scripts/idom_run.pbs

# 3) Tổng hợp kết quả: bảng mean ± std theo seed
python tools/summarize.py --csv results_step1.csv
```

`CONFIG` nhận các giá trị: `fmrec` (a), `idom_id` (b), `idom_text` (c), `idom_idtext` (d).
Các seed viết cách nhau bằng dấu `:` khi truyền qua qsub, ví dụ `SEEDS=1997:2024:42`. Đặt `PARALLEL=0` để chạy các seed lần lượt; khi đó cần khoảng gấp 3 lần walltime.

- **Walltime mặc định là 12 giờ.** Nếu queue `small_gpuq` có giới hạn thấp hơn, thêm `-l walltime=HH:MM:SS` khi qsub.
- **Đừng truyền `--gpu_id` trong job**, vì PBS đã tự đặt `CUDA_VISIBLE_DEVICES`.
- Encoder text được tải từ Hugging Face trong lần chạy đầu. Vì bước 0 chạy trên login node, thường không gặp vấn đề internet.
- Nếu copy file từ Windows sang server mà gặp lỗi `$'\r': command not found`, chạy `dos2unix scripts/*.pbs scripts/*.sh`.

## Bước 0b: đo tốc độ TRƯỚC khi chạy thật

```bash
qsub scripts/bench.pbs        # khoảng 30–60 phút; kết quả ở Ibench.o<jobid> và src/log/bench/*.json
```

Benchmark đo các con số sau cho fmrec, idom_id, idom_text và idom_idtext:
- Thời gian train mỗi epoch.
- Thời gian một lần eval trên tập val với 1, 5 và 30 bước Euler.
- Nhiễu sinh trên GPU (mặc định mới) so với trên CPU (FMRec gốc).
- Bộ nhớ GPU tối đa.

Benchmark chạy một process trên một GPU, với trọng số ngẫu nhiên (tốc độ không phụ thuộc trọng số).

Khi đã có `best.pt`, kiểm tra độ chính xác và thời gian theo số bước Euler:

```bash
python tools/eval_ckpt.py --run amazon_beauty_idom_id-text_seed1997_v2 --steps 1 2 5 10 30
```

## Checkpoint, resume, test định kỳ

- **`best.pt`** được lưu mỗi khi val cải thiện, tức sau mỗi lần eval có kết quả tốt nhất mới.
- **`last.pt`** được lưu sau mỗi lần eval, dùng để resume khi bị kill vì walltime: chỉ cần `qsub` lại đúng lệnh cũ. Khi run xong, `last.pt` bị xóa.
- **Test định kỳ:** cứ mỗi 10 epoch (`--test_interval 10`), trainer eval tập test và ghi dòng `Test epoch E` vào log. Số này chỉ để ghi lại, không dùng để chọn model.
- **Nhiễu trên GPU:** nhiễu λ được sinh trên GPU, cùng phân phối với bản gốc. Thêm `--cpu_noise` để dùng đúng cách của FMRec gốc.
- **Metric:** top-k được tính trên GPU, cho cùng kết quả với bản gốc.
- **Chọn trọng số gộp trên val:** `python tools/fuse_sweep.py --run <run>`

## Tổng hợp từ epoch tốt nhất (kể cả run đang chạy hoặc đã bị kill)

```bash
python tools/summarize_logs.py              # chọn epoch tốt nhất theo val NDCG@10 (điểm fused)
python tools/summarize_logs.py --select HR@10 --csv results_logs.csv
```

- Script đọc trực tiếp các file `src/log/<dataset>/*.out`. Với mỗi run, nó lấy epoch có val tốt nhất, rồi báo mean ± std qua các seed.
- Hàng **VAL** luôn có số liệu.
- Hàng **TEST** lấy số test cuối cùng nếu run đã xong. Nếu run chưa xong, nó lấy dòng `Test epoch E` ở đúng epoch có val tốt nhất. Trainer mới ghi dòng này mỗi 10 epoch.
- Các run chạy bằng code cũ chỉ có số VAL, cho tới khi chạy xong.

## Cách đọc kết quả

- (b) ≈ (a): việc dùng chung vector field cộng modality embedding không làm hỏng FMRec.
- So (d).fused với (a): đây là lợi ích của việc sinh song song trong không gian text.
- So (d).id với (a): nếu (d).id cao hơn, nghĩa là quỹ đạo text giúp được cả vector field dùng chung.
- So (d).text với (c): cho biết huấn luyện chung có lợi hay có hại cho quỹ đạo text.
- Nếu (d).fused không vượt được max((a), (c)), thì cần intent anchor hoặc cách fusion tốt hơn. Thử thêm `--fuse_weights 1.0 0.5` trước.

## Bước tiếp theo (chưa làm)

- Thêm intent anchor trên các x̂₁(t) của từng modality.
- Thêm baseline "FMRec + text làm condition" (một quỹ đạo duy nhất, kiểu PrismRec) để bảo vệ đóng góp "sinh song song".
