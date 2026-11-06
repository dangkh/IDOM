# IDOM — hướng dẫn chạy thí nghiệm (handoff cho phiên làm việc khác)

> Cập nhật: 06/11/2026, 22:45 (giờ Sydney). Tài liệu tự đủ: đọc file này là có thể tiếp tục chạy, không cần lịch sử chat.
> Spec nghiên cứu: `exp_spec_revise.md` (bản hiện hành) và `Exp_spec.md` (bản gốc). Báo cáo: `docs/IDOM_idea_review.pdf`.

---

## 1. Bối cảnh trong 30 giây

- **IDOM:** recommender tuần tự đa modality (ID + text; ảnh để sau), xây trên **FMRec** (flow matching, IJCAI'25, https://github.com/FengLiu-1/FMRec).
- **Dataset:** Amazon Beauty theo bản phát hành của FMRec (22.363 user, 12.101 item; trùng với Beauty.txt của S3Rec). Chia leave-one-out, xếp hạng trên toàn bộ item, metric HR/NDCG @5/10/20 (đơn vị %).
- **Hướng hiện tại (IDOM-SC):** encoder lịch sử chạy một lần mỗi batch, cộng một đầu dự đoán nhẹ cho từng bước, có tùy chọn score coupling giữa các modality (`--model eh`).
- **Ưu tiên:** đánh giá **tính khả thi** trước, nên chỉ chạy **1 seed (1997)**. Tốc độ là claim quan trọng, cần đo trước khi chạy dài.

---

## 2. Máy chủ và môi trường

| Mục | Giá trị |
|---|---|
| Hệ thống | PBS Pro (`qsub`, `qstat`, `qdel`), login node `hpc-login02` |
| Queue | `small_gpuq`, `select=1:ncpus=4:ngpus=1:mem=32gb`, walltime tối đa đã dùng 12 giờ |
| Giới hạn | **Chỉ 2 job IDOM chạy đồng thời.** Gộp nhiều run vào một job, chạy lần lượt (`scripts/queue_run.pbs`) |
| Conda | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate Astar` |
| Phiên bản trong `Astar` | torch 2.5.1+cu121, `transformers==4.46.3`, `sentence-transformers==3.3.1` (đã hạ từ 5.x/6.x vì 5.x đòi torch ≥ 2.6) |
| Repo trên server | `~/ihpc_data/hkieu/IDOM` (= `/shared/homes/u26010963/ihpc_data/hkieu/IDOM`) |
| Repo trên máy Windows | `C:\Users\26010963\Desktop\LocalDoc\IDOM`, GitHub `dangkh/IDOM` (nhánh `main`) |
| Dữ liệu raw Amazon 2014 | `~/ihpc_data/hkieu/amazon2014/{reviews_Beauty_5.json.gz, meta_Beauty.json.gz}` |

**Quy tắc vận hành:**
- Bước chỉ cần CPU (ví dụ tạo text embedding, `eval_ckpt` với ít bước) thì **chạy thẳng trên login node**, không qua queue.
- Mỗi run chiếm trọn 1 GPU. Không chạy nhiều seed song song trong một job, vì 3 process tranh nhau 4 CPU và mỗi process chậm 2–3,5 lần.
- Đồng bộ code lên server bằng `git push` / `git pull`. Nếu GitHub báo `remote: fatal error in commit_refs`, copy file bằng `scp` rồi push sau.
- Nếu gặp lỗi `$'\r': command not found` (file copy từ Windows), chạy `dos2unix scripts/*.pbs scripts/*.sh`.

---

## 3. Cấu trúc repo

```
src/      main.py  config.py (mọi tham số)  model.py (FMRec, IDOM, build_model)  eh_model.py (IDOM-SC)
          fmrec.py (flow của FMRec)  trainer.py (train/eval/checkpoint/resume)  utils.py (dataloader)
scripts/  idom_run.pbs (1 cấu hình, các seed)  queue_run.pbs (nhiều run lần lượt trong 1 job)  submit_all.sh
          smoke_eh.pbs  bench.pbs  eval_steps.pbs  prepare_text.sh  prepare_beauty_text.py
tools/    summarize.py  summarize_logs.py  summarize_steps.py  eval_ckpt.py  fuse_sweep.py
          ensemble_eval.py  benchmark_speed.py  smoke_eh.py
datasets/data/amazon_beauty/  dataset.pkl  text_emb_all-MiniLM-L6-v2.npy (12.102 × 384, hàng 0 = padding)
```

- **Tên run:** `<dataset>_<model>_<modalities>_seed<seed>[_<tag>]`, ví dụ `amazon_beauty_eh_id-text_seed1997_eh0`. Với `fmrec`, phần modalities là `id`.
- **Log:** `src/log/amazon_beauty/<run>.out` (stdout của từng run), `<run>_<thời gian>.log`, `<run>_<thời gian>.json` (kết quả cuối, chỉ có khi run xong).
- **Checkpoint:** `src/ckpt/amazon_beauty/<run>/best.pt` (lưu mỗi khi val cải thiện; có `args`) và `last.pt` (để resume; bị xóa khi run xong).

---

## 4. Mặc định quan trọng (`src/config.py`)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `--sample_N` | 5 | số bước Euler khi suy luận (FMRec gốc dùng 30; đã đo 1 = 5 = 30) |
| `--eval_interval` / `--test_interval` | 20 / 20 | các cấu hình eh dùng 10 / 10 |
| `--patience` | 5 | số lần eval không cải thiện thì dừng |
| `--epochs` | 500 | cấu hình eh: 100 (fine-tune) |
| `--lr` / `--decay_step` / `--gamma` | 1e-3 / 100 / 0,1 | |
| `--Loss_Alpha` / `--Loss_Beta` | 0,2 / 0,4 | CE / decoder phụ của FMRec |
| `--resume` | (job luôn bật) | chạy tiếp từ `last.pt` nếu có |
| Nhiễu λ của FMRec | sinh trên GPU | `--cpu_noise` = cách gốc (chậm) |
| Chọn model | lưu khi **bất kỳ** metric val nào cải thiện | giống FMRec |

**Riêng `--model eh`:** `--head_off` (B4), `--head_layers 1`, `--head_heads 4`, `--K_train` (0 = không rollout), `--coupling`, `--x_frozen` (C2), `--q_const` (C6), `--msg_off_eval` (C5), `--grad_step0`, `--lam_flow 1.0`, `--lam_rank 0.2`, `--lam_branch 0.1`, `--tau 1.0`, `--warm_start <best.pt>`, `--lr_pretrained_scale 0.1`.

---

## 5. Các cấu hình job (`scripts/idom_run.pbs`, tham số `CONFIG`)

| CONFIG | Tag | Ý nghĩa | Khởi tạo từ | Epoch |
|---|---|---|---|---|
| `fmrec` | — | FMRec gốc, chỉ ID | — | 500 |
| `idom_id` / `idom_text` / `idom_idtext` | — | IDOM: chỉ ID / chỉ text / ID + text | — | 500 |
| `eh_b4` | `b4` | B4: head tắt, dự đoán trực tiếp (không x_t) | IDOM v3 cùng seed | 100 |
| `eh0` | `eh0` | EH-0: encoder + head, chưa coupling | IDOM v3 | 100 |
| `eh0_nodec` | `eh0nodec` | EH-0 bỏ decoder phụ | IDOM v3 | 100 |
| `c0` | `c0` | rollout K = 2, không message | eh0 | 100 |
| `c2` | `c2` | lặp tinh chỉnh không ODE (x = 0) + message | eh0 | 100 |
| `c4` | `c4` | coupling đầy đủ (đề xuất) | eh0 | 100 |
| `c6` | `c6` | c4 với intent = 0 | eh0 | 100 |

- Biến môi trường khác: `SEEDS`, `EPOCHS`, `TAG` (nối vào sau tag của cấu hình), `EXTRA_ARGS` (không chứa dấu phẩy hay khoảng trắng; ví dụ `--test_interval=20`), `IDOM_TAG` (mặc định `v3`), `EH_TAG` (mặc định `eh0`), `CONDA_ENV`.
- Job tự **bỏ qua** run đã có JSON kết quả, **resume** run đang dở, và **từ chối** chạy nếu thiếu checkpoint khởi tạo.

**Mẫu lệnh:**

```bash
cd ~/ihpc_data/hkieu/IDOM
# nhiều run lần lượt trong 1 job (cách nên dùng, vì quota 2 job)
qsub -N IqF -v RUNS="eh_b4/1997+eh0/1997" scripts/queue_run.pbs
# 1 cấu hình, 1 seed
qsub -v CONFIG=c4,SEEDS=1997 scripts/idom_run.pbs
# job bị kill vì walltime -> gửi lại ĐÚNG lệnh cũ (resume từ checkpoint)
```

---

## 6. Công cụ

| Lệnh | Mục đích |
|---|---|
| `qsub scripts/smoke_eh.pbs` | **Bắt buộc trước khi train eh.** 6 kiểm tra đúng đắn + đo tốc độ eh so với IDOM (output ở `Ismoke.o<jobid>`) |
| `python tools/summarize.py` | Bảng test mean ± std từ các file JSON (run đã xong) |
| `python tools/summarize_logs.py [--select HR@10]` | Epoch tốt nhất theo val, kể cả run đang chạy hoặc bị kill (lấy số test ở cùng epoch) |
| `qsub -v RUNS="<run>+<run>",STEPS="1:2:3:5" scripts/eval_steps.pbs` | Độ chính xác và thời gian theo số bước Euler (GPU) |
| `python tools/summarize_steps.py` | Gộp các file `*_steps.json` |
| `python tools/eval_ckpt.py --run <run> --steps 1 2 5 [--msg_off]` | Eval một checkpoint (`--msg_off` = C5) |
| `python tools/ensemble_eval.py --a <run> --b <run> [--a_key id --b_key text]` | B2: gộp hai checkpoint, chọn trọng số trên val, bảng rescue/harm |
| `python tools/fuse_sweep.py --run <run>` | Chọn trọng số gộp ID/text của một model 2 modality trên val |
| `qsub scripts/bench.pbs` | Benchmark tốc độ (train s/epoch, eval theo số bước, bộ nhớ GPU) |
| `qstat -x -u $USER`, `qstat -xf <jobid> \| grep Exit_status` | Trạng thái job, kể cả job đã xong |

---

## 7. Kết quả đã có (test, %, Beauty)

| Model | Seeds | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Ghi chú |
|---|---|---|---|---|---|---|
| FMRec (bài báo) | — | 8,27 | 4,95 | 11,63 | 5,79 | |
| FMRec v3 | 1997, 2024, 42 | 8,36 ± 0,06 | 4,97 ± 0,02 | 11,71 ± 0,03 | 5,81 ± 0,03 | `amazon_beauty_fmrec_id_seed*_v3` |
| IDOM chỉ ID (code cũ) | 3 | 8,30 ± 0,06 | 4,96 ± 0,04 | 11,81 ± 0,10 | 5,84 ± 0,05 | sanity, không có checkpoint |
| IDOM ID + text v3: điểm gộp | 3 | **9,18 ± 0,10** | **5,21 ± 0,07** | **13,11 ± 0,12** | **6,20 ± 0,08** | `amazon_beauty_idom_id-text_seed*_v3` |
| … nhánh ID | 3 | 8,41 ± 0,07 | 4,93 ± 0,07 | 11,82 ± 0,10 | 5,79 ± 0,08 | |
| … nhánh text | 3 | 7,02 ± 0,06 | 3,53 ± 0,04 | 10,93 ± 0,10 | 4,51 ± 0,05 | |

**Các phát hiện khác:**
- **Số bước Euler (IDOM ID + text v3):** 1, 2, 5, 10, 30 bước cho cùng kết quả (chênh ≤ 0,01 điểm). Thời gian một lượt test: 4,4 s / 8,0 s / 19,3 s / 37,0 s / 109,1 s. Nguyên nhân: bộ lấy mẫu của FMRec chỉ lấy trung bình các dự đoán, và mạng gần như bỏ qua x_t (hệ số λ ≈ 0,001).
- **Text-only train riêng** (code cũ, chỉ có số val): HR@10 9,76, NDCG@10 5,23. Nhánh text trong IDOM chỉ đạt 9,19 / 4,81 trên val, tức có thể bị ảnh hưởng tiêu cực khi train chung.
- **Tốc độ** (benchmark, một process): FMRec 34,2 s/epoch, eval 1 bước 1,2 s / 30 bước 47,1 s, 2,6 GB. IDOM ID + text 62,3 s/epoch, eval 1 bước 4,1 s / 30 bước 91,5 s, 4,9 GB. Trên node nhanh, FMRec v3 chạy 17,5 s/epoch.
- **Thời gian train thuần** (v3): FMRec khoảng 1,9 giờ (dừng ở epoch 300–440); IDOM khoảng 4,8 giờ (dừng ở epoch 420–500).

---

## 8. Trạng thái hiện tại và việc tiếp theo

**Đang chạy hoặc vừa gửi:**
- IqD: text-only v3 (`idom_text/1997+2024+42`, `TAG=v3`), để có checkpoint cho baseline B2.

**Code mới (eh, ensemble, smoke) đã viết nhưng CHƯA chạy lần nào.**

**Thứ tự cần làm** (chi tiết và điều kiện dừng/đi tiếp ở `exp_spec_revise.md` §6):

1. `qsub scripts/smoke_eh.pbs` → mọi test phải PASS; ghi lại tỉ lệ tốc độ eh0 và eh_c4 so với IDOM. Mục tiêu: eh0 ≤ 1,2× IDOM.
2. Khi IqD xong:
   ```bash
   python tools/ensemble_eval.py --a amazon_beauty_fmrec_id_seed1997_v3 --b amazon_beauty_idom_text_seed1997_v3
   python tools/ensemble_eval.py --a amazon_beauty_idom_id-text_seed1997_v3 --a_key id --b amazon_beauty_idom_id-text_seed1997_v3 --b_key text
   ```
   **Điều kiện:** IDOM (đã tune trọng số gộp) phải ≥ ensemble B2.
3. `qsub -N IqF -v RUNS="eh_b4/1997+eh0/1997" scripts/queue_run.pbs` (và tùy chọn `qsub -N IqG -v RUNS="eh0_nodec/1997" scripts/queue_run.pbs`).
   Sau đó: `qsub -v RUNS="amazon_beauty_eh_id-text_seed1997_eh0",STEPS="1:2:3:5" scripts/eval_steps.pbs`.
   **Điều kiện:** eh0 ≥ IDOM v3, và số bước K có tạo khác biệt. Nếu K vẫn không có tác dụng, bỏ claim ODE và chuyển framing sang "iterative score refinement".
4. Chỉ khi bước 3 đạt:
   ```bash
   qsub -N IqH -v RUNS="c0/1997+c2/1997" scripts/queue_run.pbs
   qsub -N IqI -v RUNS="c4/1997+c6/1997" scripts/queue_run.pbs
   python tools/eval_ckpt.py --run amazon_beauty_eh_id-text_seed1997_c4 --steps 2 --msg_off   # C5
   ```
   **Điều kiện:** C4 > C0 và C4 > B2. C4 > C2 là điều kiện để claim ODE.
5. Chỉ chạy 3 seed cho cấu hình thắng sau khi giai đoạn 2 đạt.

---

## 9. Những điều cần nhớ

- **Không claim "IDOM nhanh hơn FMRec"**: FMRec cũng chạy được 1 bước mà không mất độ chính xác. **FAVE** (arXiv 2604.04427) đã phê phán việc lấy mẫu nhiều bước; chưa xác nhận được venue SIGIR 2026.
- **Gộp điểm hiện tại** là tổng log-softmax với trọng số 1:1, cho cùng thứ hạng với tổng điểm thô, tức là **chưa hiệu chỉnh**.
- **Với bộ lấy mẫu chuẩn** v = (x̂₁ − x)/(1 − t), điểm cuối x_K luôn bằng x̂₁ ở bước cuối. Vì vậy feedback phải đi vào **đầu dự đoán**; cộng vào vận tốc thì feedback ở các bước trước bị xóa.
- **Mỗi run IDOM-SC** đều fine-tune 100 epoch từ checkpoint cùng seed. Các đối chứng (C0, C2, C5, C6) phải có cùng ngân sách để so sánh công bằng.
- **Text embedding:** `scripts/prepare_text.sh <thư mục raw>` (CPU, login node). Script dựng lại mapping ID → ASIN từ file raw 2014 và kiểm tra khớp 22.363/22.363 trước khi ghi.
