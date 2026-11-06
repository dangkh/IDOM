# IDOM-SC (revised): Cached-Encoder Score-Coupled Flow for Multimodal Sequential Recommendation

> **Bản sửa của `Exp_spec.md`**  
> **Ngày:** 06/11/2026  
> **Trạng thái:** Đề xuất, **chưa có thực nghiệm**. Mọi con số về chi phí trong tài liệu này là **ước tính**, cần đo lại bằng `tools/benchmark_speed.py` sau khi có code.  
> **Thay đổi chính:** (1) tách mạng thành *encoder lịch sử chạy một lần* + *đầu dự đoán nhẹ cho từng bước*; (2) message đi vào **đầu dự đoán**, không chỉ cộng vào vận tốc; (3) message **không chuẩn hóa** theo tổng trọng số; (4) lộ trình được rút gọn cho quota 2 job GPU.  
> **Các phần giữ nguyên** từ `Exp_spec.md`: giả thuyết H1–H5 (§2), checklist chống rò rỉ nhãn (§8.3), metrics và diagnostics (§9), failure modes (§11), novelty boundary (§13), câu hỏi mở (§15). Tài liệu này chỉ viết lại những phần thay đổi.

---

## 0. Tóm tắt thay đổi so với `Exp_spec.md`

| # | Mục trong bản cũ | Bản cũ | Bản sửa | Lý do |
|---|---|---|---|---|
| 1 | §3.1, §4.5 kiến trúc | Dùng nguyên mạng FMRec; transformer chạy lại toàn bộ lịch sử ở **mỗi bước** | **Encoder lịch sử chạy 1 lần/batch** (không nhận x_t) + **đầu dự đoán 1 lớp** nhận (cache lịch sử, x_t, t, message) | Giảm chi phí train từ ~2–2,5× xuống ~1,1× IDOM; làm cho trạng thái x_t thực sự ảnh hưởng tới dự đoán |
| 2 | §4.5 nơi đưa feedback | `βG` cộng vào vận tốc (velocity guidance) | Message là **một token đầu vào của đầu dự đoán**; vận tốc chỉ là `(x̂₁ − x)/(1 − t)` | Với predictor không nhạy trạng thái, feedback cộng vào vận tốc ở các bước trước **bị xóa hết** (chứng minh ở §2) |
| 3 | §4.4 công thức message | `r = Σ wᵢ eᵢ / (ε + Σ wᵢ)`, sau đó LayerNorm | `r = Σ wᵢ eᵢ` (không chia), cộng thêm đặc trưng vô hướng về mức bất đồng | Bản cũ khuếch đại nhiễu khi hai modality gần như đồng ý |
| 4 | §5.1 nhánh teacher FM | Forward riêng một lần nữa trên toàn bộ transformer | **Dùng chung encoder** với rollout; chỉ chạy thêm đầu dự đoán; có thể lấy nhiều t cho mỗi lịch sử | Chi phí gần như bằng 0 |
| 5 | §6.2 decode điểm cuối | Để ngỏ (x_K hay x̂₁) | Với sampler `v = (x̂₁ − x)/(1 − t)`: **x_K = x̂₁ ở bước cuối**, hai cách là một | Hệ quả đại số (§2.2), hết mơ hồ |
| 6 | §5.4 K khi train | Cố định 3 hoặc lấy ngẫu nhiên {2, 3, 5} | **K_train = 2** cho MVP; K_test ∈ {1, 2, 3, 5} | Giảm chi phí; đủ để có tương tác giữa các bước |
| 7 | §7, §10 lộ trình | ~11 cấu hình C0–C10 × 3 seed từ đầu (~500 giờ GPU) | Giai đoạn 0–3; **khởi tạo từ checkpoint**, **sàng lọc 1 seed**; ~12–15 run cho tới quyết định đầu tiên | Quota 2 job |
| 8 | §1.2 FMRec | Chưa kiểm chứng độc lập | Đã kiểm chứng trên IDOM: 1 bước = 30 bước (chênh ≤ 0,01 điểm, 3 seed) | Kết quả `Isteps` |
| 9 | §13 FAVE | Ghi SIGIR 2026 | Abstract trên arXiv (2604.04427) **không ghi venue**; cần xác minh trước khi trích dẫn | Kiểm tra 06/11/2026 |

---

## 1. Bằng chứng hiện có (cập nhật)

Như `Exp_spec.md` §1.1, cộng thêm:

- **Số bước Euler không ảnh hưởng tới IDOM** (ID + text, v3, 3 seed, test): điểm gộp HR@10 = 9,17 (1 bước) so với 9,18 (30 bước); thời gian một lượt test 4,4 s so với 109,1 s. Kết quả của FMRec trong cùng job còn chờ đối chiếu.
- **Chi phí đo được** (benchmark một process, nhiễu sinh trên GPU):

| Model | Train, s/epoch | Một lượt val, 1 bước | Một lượt val, 30 bước | Bộ nhớ GPU |
|---|---:|---:|---:|---:|
| FMRec | 34,2 | 1,2 s | 47,1 s | 2,6 GB |
| IDOM ID + text | 62,3 | 4,1 s | 91,5 s | 4,9 GB |

- **Fusion hiện tại** `Σ log_softmax(s_m)` với trọng số 1:1 cho cùng thứ hạng với tổng điểm thô `Σ s_m` (§4.1 bản cũ). Kết quả +9,8% HR@10 vì vậy là của fusion **chưa hiệu chỉnh**.

---

## 2. Vì sao phải đổi kiến trúc (lập luận cho thay đổi 1, 2, 5)

### 2.1. FMRec mã hóa lại lịch sử ở mỗi bước

Trong FMRec, x_t và t được cộng vào **mọi vị trí** của chuỗi lịch sử (`rep + λ ⊙ (x_t + emb_t)`), nên mỗi bước Euler phải chạy lại toàn bộ transformer 4 lớp trên 50 item. Lịch sử không đổi giữa các bước, nên đây là phần tính toán lặp lại. Đồng thời, hệ số λ ≈ 0,001 khiến mạng gần như bỏ qua x_t và t.

### 2.2. Với sampler chuẩn, điểm cuối luôn bằng dự đoán ở bước cuối

Với t_k = k/K, Δt = 1/K và `x_{k+1} = x_k + Δt·(x̂₁,k − x_k)/(1 − t_k)`:

```
x_{k+1} − x̂₁,k = (x_k − x̂₁,k) · (K − k − 1)/(K − k)
```

Ở bước cuối (k = K − 1), hệ số bằng 0, nên **x_K = x̂₁,K−1**.

Hệ quả:

1. **Decode bằng x_K hay bằng x̂₁ ở bước cuối là một.** Thay đổi 5 trong bảng.
2. **Nếu feedback chỉ được cộng vào vận tốc** (`+ Δt·β·G_k`), mọi phần cộng ở các bước k < K − 1 bị hệ số trên triệt tiêu. Chỉ còn lại `Δt·β·G_{K−1}`. Cơ chế suy biến thành "feedback ở điểm cuối" (C1).
3. **Ảnh hưởng giữa các bước chỉ có thể đi qua x̂₁.** x̂₁,k phải phụ thuộc vào x_k (trạng thái) và vào message. Nếu x̂₁ chỉ phụ thuộc message mà không phụ thuộc trạng thái, mô hình là **lặp tinh chỉnh không ODE** (C2) theo đúng cấu trúc, và giả thuyết H3 không thể đúng.

→ Điều kiện tiên quyết của IDOM-SC: **đầu dự đoán phải nhạy với x_t** (giai đoạn 1 kiểm tra), và **message phải đi vào đầu dự đoán**.

---

## 3. Kiến trúc mới: encoder lịch sử + đầu dự đoán theo bước

### 3.1. Thành phần

```
H_u (50 item) ──► HistoryEncoder (transformer 4 lớp, KHÔNG nhận x_t, t)  ──► cache: S ∈ R^{L×d}, h_last ∈ R^d
                                                                              │ (chạy 1 lần / batch / modality)
mỗi bước k, mỗi modality m:
    query token  = Embed_x(x_k^m) + emb(m)
    message token = W_msg(r_m,k) + MLP(scalars_m,k)          (zero-init W_msg)
    StepHead_m: 1 lớp, [query, message] cross-attend vào S^m (K/V của S cache sẵn), AdaLN theo t_k
    x̂₁,k^m = h_last^m + Head_out(...)                         (zero-init Head_out)
```

- **HistoryEncoder:** dùng lại transformer của FMRec/IDOM, bỏ hoàn toàn phép cộng `λ ⊙ (x_t + emb_t)`. Encoder dùng chung giữa các modality như IDOM (thêm modality embedding), hoặc tách riêng ở ablation B5.
- **StepHead:** 1 lớp gồm cross-attention của 2 token (query, message) vào lịch sử đã cache, FFN 4d, và AdaLN theo `emb(t)`. K/V của lịch sử được chiếu **một lần** và cache lại.
- **Residual quanh `h_last`, zero-init lớp ra:** lúc khởi tạo, x̂₁ = h_last, đúng bằng bộ dự đoán một bước không dùng x_t (baseline B4). Mô hình bắt đầu từ một baseline đã biết, và mọi cải thiện đến từ đầu dự đoán.
- **Bảng item theo modality** `E^m` (ID: embedding học được; text: `text_proj(e_text)`) được tính **một lần mỗi batch**.

### 3.2. Sampler

`v_k^m = (x̂₁,k^m − x_k^m)/(1 − t_k)`, `x_{k+1}^m = x_k^m + Δt·v_k^m`, x_0^m = z_0^m ~ N(0, I) độc lập giữa các modality. Điểm cuối `s^m = x̂₁,K−1^m · (E^m)ᵀ` (tương đương x_K theo §2.2).

### 3.3. Ước tính chi phí (cần đo lại)

Số phép nhân-cộng cho mỗi hàng (d = 128, L = 50, V = 12.101): encoder ~42M, decoder phụ ~26M, StepHead ~0,2M, tính điểm ~1,5M mỗi modality, message ~1,5M mỗi cặp modality.

| Cấu hình | Train, so với IDOM hiện tại | Một lượt eval | Ghi chú |
|---|---:|---:|---|
| IDOM hiện tại | 1× (62 s/epoch) | 4,1 s (1 bước) | đo được |
| IDOM-SC kiến trúc cũ, K = 2 | ~2–2,5× | ~9–11 s | ước tính ở thảo luận trước |
| **Encoder + head, chưa coupling, K = 2** | **~1,0–1,1×** | **~4,5 s** | ước tính |
| **Encoder + head + coupling, K = 2** | **~1,1–1,2×** | **~5 s** | ước tính |
| Như trên, bỏ decoder phụ (`Loss_Beta = 0`) | ~0,7–0,8× | ~5 s | cần ablation độ chính xác |

---

## 4. Score coupling (sửa từ `Exp_spec.md` §4)

Giữ nguyên §4.1 (điểm theo modality, softmax với τ_m), §4.2 (positive discrepancy là tín hiệu đề xuất, không phải bằng chứng complementarity), §4.6 (cập nhật đồng bộ). Sửa:

### 4.1. Message không chuẩn hóa

```
w_{n→m,k}(i) = α_{n→m,k}(u) · [p_k^n(i) − p_k^m(i)]_+
r_{n→m,k}    = Σ_i w_{n→m,k}(i) · e_i^m                      (không chia cho Σ w)
D_{n→m,k}    = Σ_i [p_k^n(i) − p_k^m(i)]_+  ∈ [0, 1]           (= total variation distance)
```

- Độ lớn của r tự phản ánh mức bất đồng: khi hai modality đồng ý, r ≈ 0.
- Đưa các đại lượng vô hướng `scalars_m,k = [D, entropy(p^n), entropy(p^m), margin top1–top2]` vào message token qua một MLP nhỏ.
- **Ablation:** bản chuẩn hóa của spec cũ, để đối chiếu.

### 4.2. Gate theo user

Giữ như §4.3 bản cũ: `α = sigmoid(g(q_u, summary(p^n), summary(p^m), t_k, emb(n), emb(m)))`, với `q_u = LN(MLP(h_last^ID))`.

Ghi rõ trong bài: q_u là **biểu diễn lịch sử**. Chỉ gọi nó là "intent" khi các ablation C6/C7 cho thấy nó có vai trò riêng.

### 4.3. Đưa message vào đầu dự đoán

Message token đi vào StepHead (§3.1), **không** cộng vào vận tốc. Có zero-init để lúc khởi tạo mô hình đúng bằng bản không coupling. Không còn tham số β; độ mạnh của feedback do W_msg học. Ablation "β = 0" (C5) trở thành "message token = 0".

---

## 5. Training protocol (sửa từ `Exp_spec.md` §5)

### 5.1. Một lần encoder, hai nhánh loss

```python
S, h_last = encode(history)                 # mỗi modality; 1 lần / batch
E = {m: item_table(m) for m in mods}         # 1 lần / batch

# Nhánh A: rollout tự sinh, giống hệt lúc suy luận (không có target trong input)
x = {m: randn(B, d) for m in mods}
for k in range(K_train):                    # K_train = 2
    t = k / K_train
    msg = coupler(xhat_prev_scores, ...) if k > 0 else zeros   # bước 0 chưa có điểm số trước đó
    xhat = {m: head(m, S[m], h_last[m], x[m], t, msg[m]) for m in mods}
    scores = {m: xhat[m] @ E[m].T for m in mods}
    x = {m: x[m] + (xhat[m] - x[m]) / K_train / (1 - t) for m in mods}   # cập nhật đồng bộ
L_rank = CE(fuse(scores), y) + lam_branch * sum(CE(scores[m], y) for m in mods)

# Nhánh B: teacher flow matching, CHỈ chạy head, dùng lại S và h_last
t_fm = sample_t(B, n_t)                     # có thể n_t > 1 mẫu t cho mỗi lịch sử, gần như miễn phí
x_t = t_fm * E[m][y] + (1 - t_fm) * z0
L_FM = sum(MSE(head(m, S[m], h_last[m], x_t, t_fm, msg=0), E[m][y]) for m in mods)

L = L_rank + lam_flow * L_FM (+ lam_dec * L_aux_decoder, nếu giữ)
```

**Lưu ý về bước 0:** message ở bước k được tính từ điểm số của bước k − 1. Theo thiết kế "message tác động bước tiếp theo" của spec cũ, ở bước 0 message bằng 0. Vì vậy:
- Với K = 1 không có coupling.
- Với K = 2, coupling chỉ tác động một lần (từ bước 0 sang bước 1). Đây là MVP tối thiểu.
- K_test = 3, 5 dùng để kiểm tra khả năng tổng quát theo số bước.

### 5.2. Các tùy chọn tiết kiệm (mỗi tùy chọn phải được báo cáo)

1. **Không backprop qua bước 0** (`no_grad` ở bước 0, giữ gradient qua message của bước 1). Ablation: backprop đầy đủ.
2. **bf16 autocast.** Kiểm tra trên 1 seed rằng độ chính xác không đổi.
3. **Bỏ decoder phụ** (`Loss_Beta = 0`): ablation riêng trên kiến trúc encoder + head, chưa có coupling.

### 5.3. Khởi tạo từ checkpoint

- Encoder, bảng item, `text_proj` lấy từ `best.pt` của IDOM v3 (cùng seed). Phần x_t-injection cũ bị bỏ, nên trọng số transformer dùng lại được trực tiếp.
- StepHead và coupler khởi tạo mới, với lớp ra bằng 0.
- Fine-tune tối đa 100 epoch, eval mỗi 10 epoch, patience 5.
- **Mọi cấu hình so sánh** (kể cả C0, C2, C5) phải fine-tune cùng số epoch từ cùng checkpoint.
- Một run train từ đầu cho cấu hình thắng cuối cùng, để chắc chắn kết quả không phụ thuộc vào việc khởi tạo.

---

## 6. Lộ trình mới và điều kiện dừng/đi tiếp

Thời gian ước tính trên node nhanh (như IqB); trên node chậm nhân khoảng 2. Quota: 2 job, chạy lần lượt bằng `scripts/queue_run.pbs`.

### Giai đoạn 0: baseline còn thiếu (một phần đang chạy)

| ID | Run | Số run | Thời gian ước tính |
|---|---|---:|---|
| B1 | text-only v3, 3 seed (IqD, đang chạy) | 3 | ~1,5–3 giờ/seed |
| B2 | ensemble FMRec v3 + text-only v3, tune trọng số và τ trên val | 0 (chỉ eval) | vài phút |
| B3' | IDOM v3, tune trọng số gộp trên val (`fuse_sweep.py`) | 0 (chỉ eval) | vài phút |
| B4 | encoder + head với **head tắt** (x̂₁ = h_last), tức bộ dự đoán không dùng x_t | 1 seed | ~2 giờ (warm start) |
| diag | Độ nhạy của x̂₁ theo x_t, t trên checkpoint IDOM v3 | 0 (chỉ eval) | vài phút |

**Điều kiện đi tiếp:** IDOM (đã tune fusion) ≥ ensemble B2. Nếu ensemble thắng rõ, IDOM-SC phải được chứng minh vượt chính B2, và cần đánh giá lại xem có đáng tiếp tục không.

### Giai đoạn 1: làm cho trạng thái có ý nghĩa (chưa coupling)

| ID | Run | Số run |
|---|---|---:|
| EH-0 | Encoder + head, teacher FM + CE, ID + text, K_test ∈ {1, 2, 3, 5} | 1 seed |
| EH-0b | Như EH-0, bỏ decoder phụ | 1 seed |

**Điều kiện đi tiếp** (cả ba phải đạt):
1. Độ chính xác ở K tốt nhất ≥ IDOM v3 (đã tune fusion).
2. **K có tác dụng:** kết quả K > 1 khác K = 1 vượt mức dao động giữa các seed, hoặc ít nhất độ nhạy của x̂₁ theo x_t, t khác 0 rõ rệt.
3. Chi phí đo được ≤ 1,2× IDOM mỗi epoch.

Nếu điều kiện 2 không đạt: mạng vẫn bỏ qua trạng thái dù đã đưa x_t vào token riêng. Khi đó đổi framing thành "iterative score refinement", bỏ claim ODE, và chạy thẳng C2/C4 dưới dạng không ODE.

### Giai đoạn 2: coupling MVP (ID + text, K_train = 2), khởi tạo từ EH-0

| ID | Cấu hình | Mục đích |
|---|---|---|
| C0 | EH-0 fine-tune thêm, không message | đối chứng chính, cùng ngân sách |
| C2 | Lặp tinh chỉnh không ODE: cùng head, cùng message, nhưng **x cố định bằng 0** qua các bước (chỉ message thay đổi) | phân biệt lợi ích ODE với lợi ích lặp |
| C4 | Coupling đầy đủ (đề xuất) | — |
| C5 | C4 với message token = 0 khi suy luận | xác nhận message có tác dụng |
| C6 | C4 với q_u = hằng số | vai trò của "intent" |

5 run × 1 seed × ~2–3 giờ (fine-tune 100 epoch). Với 2 job song song, khoảng 6–8 giờ.

**Điều kiện đi tiếp:** C4 > C0, C4 > B2, và net rescue (§9.2 bản cũ) của C4 > 0, tất cả trên val. C4 > C2 là điều kiện **để claim ODE**. Nếu C4 ≈ C2, đổi framing thành iterative refinement như §10 bản cũ.

### Giai đoạn 3: xác nhận (chỉ khi giai đoạn 2 đạt)

- 3 seed cho C0, C2, C4 và B2, kèm paired bootstrap 95% CI.
- Một run train từ đầu cho C4.
- Ablation C7–C10 của bản cũ, message chuẩn hóa so với không chuẩn hóa, backprop đầy đủ so với no_grad bước 0.
- Thêm ảnh: Amazon 2014 có file `image_features_Beauty.b` (đặc trưng CNN 4096 chiều theo ASIN). Mapping ASIN đã được dựng lại và kiểm chứng, nhưng cần kiểm tra độ phủ trước khi dùng.
- Dataset thứ hai (Sports hoặc Toys) với cùng protocol.

Tổng số run tới quyết định ở cuối giai đoạn 2: khoảng **10–12 run**, so với khoảng 33+ run của bản cũ.

---

## 7. Bảng baseline (thay `Exp_spec.md` §7.1–7.2)

| ID | Baseline/control | Trạng thái |
|---|---|---|
| B0 | FMRec v3, 3 seed, K ∈ {1, 30} | đã có (số theo bước chờ đối chiếu) |
| B1 | text-only, train riêng | đang chạy (IqD) |
| B2 | ensemble B0 + B1, tune trên val | chờ B1 |
| B3 | IDOM v3, fusion 1:1 / đã tune | đã có / chờ `fuse_sweep` |
| B4 | encoder + head tắt (dự đoán trực tiếp, không x_t) | giai đoạn 0 |
| B5 | encoder dùng chung so với tách riêng theo modality | giai đoạn 3 |
| EH-0 | encoder + head, chưa coupling | giai đoạn 1 |
| C0, C2, C4, C5, C6 | như §6 | giai đoạn 2 |
| C7–C10 | như bản cũ | giai đoạn 3 |

---

## 8. Code cần viết (thay `Exp_spec.md` §14)

| File | Nội dung |
|---|---|
| `src/eh_model.py` | `HistoryEncoder` (dùng lại transformer, bỏ x_t-injection, trả về S và h_last, cache K/V); `StepHead` (2 token, cross-attention, AdaLN theo t, zero-init); `EHModel` với `compute_loss` / `predict_scores(K)` theo interface của trainer hiện tại |
| `src/score_coupler.py` | calibrate → positive discrepancy → gate → `Σ w·E_receiver` (không chuẩn hóa) + scalars |
| `src/coupled_solver.py` | rollout K bước, cập nhật đồng bộ, các cờ: `no_grad_step0`, `msg_off`, `x_frozen` (cho C2), `q_const` |
| `src/config.py` | thêm `--model eh`, `--head_layers`, `--K_train`, `--coupling`, `--warm_start <ckpt>`, `--amp` |
| `tools/benchmark_speed.py` | thêm cấu hình `eh`, `eh_coupled` |
| `tools/ensemble_eval.py` | B2: gộp hai checkpoint, tune trọng số và τ trên val, bảng rescue/harm |
| `tests/` | (1) head zero-init cho x̂₁ = h_last; (2) sampler cho x_K = x̂₁ ở bước cuối; (3) rollout không đọc target; (4) cập nhật đồng bộ; (5) message = 0 khi hai phân phối bằng nhau; (6) gradient đi được từ loss qua message tới modality nguồn |

---

## 9. Mẫu báo cáo (bổ sung vào `Exp_spec.md` §12)

Thêm các trường sau vào YAML của mỗi run:

```yaml
architecture: "encoder_head"        # hoặc "fmrec_full"
head_layers: 1
warm_start: "ckpt/amazon_beauty/amazon_beauty_idom_id-text_seed1997_v3/best.pt"
finetune_epochs: 100
K_train: 2
no_grad_step0: true
amp: "bf16"
aux_decoder: true                   # Loss_Beta > 0
message_normalized: false
x_frozen: false                     # true = C2
state_sensitivity:                  # đo trên val, cùng H và z0
  dxhat_dt: null                    # ||x̂₁(t=0.9) − x̂₁(t=0.1)|| / ||x̂₁||
  dxhat_dz0: null                   # độ lệch của x̂₁ khi đổi z0
```

---

## 10. Rủi ro riêng của kiến trúc mới

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Head 1 lớp quá yếu, độ chính xác giảm so với IDOM | EH-0 < IDOM v3 ở mọi K | Tăng lên 2 lớp; hoặc cho head thêm self-attention giữa query và message |
| Mạng học cách chép x_t khi t gần 1 | x̂₁ ≈ x_t ở t lớn; K lớn tốt hơn hẳn K nhỏ khi train, nhưng không khi test | Kiểm tra phân phối t khi train; thử v-prediction (phương án B của spec cũ) |
| Kết quả phụ thuộc vào z₀ | Độ lệch chuẩn HR@10 qua nhiều z₀ lớn hơn qua các seed | Báo cáo trung bình qua nhiều z₀; thử z₀ = 0 khi suy luận |
| Khởi tạo từ checkpoint tạo lợi thế không công bằng | Cấu hình được fine-tune tốt hơn cấu hình train từ đầu | Mọi đối chứng fine-tune như nhau; một run từ đầu cho cấu hình thắng |
| Lệch phân phối: nhánh FM train với message = 0, suy luận có message | Message làm giảm độ chính xác ở C4 so với C0 | Thêm một phần nhánh FM có message lấy từ rollout (đã detach) |

---

## 11. Thứ tự thực hiện thực tế

1. Chờ IqD (text-only v3) xong → viết `ensemble_eval.py` → có B2 và B3' (tune fusion).
2. Viết `eh_model.py` + test 1–2 → benchmark tốc độ (khoảng 10 phút) → xác nhận ≤ 1,2× IDOM.
3. Chạy B4 và EH-0 (2 job song song) → kiểm tra điều kiện của giai đoạn 1.
4. Nếu đạt: viết `score_coupler.py` và `coupled_solver.py` + test 3–6 → chạy C0, C2, C4, C5, C6.
5. Quyết định theo điều kiện của giai đoạn 2: **giữ ODE** / **đổi thành iterative refinement** / **dừng**.
