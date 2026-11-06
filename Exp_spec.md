# IDOM-SC: Intent-Conditioned Score-Coupled ODE for Multimodal Sequential Recommendation

> **Experimental specification / research handoff**  
> **Ngày:** 06/11/2026  
> **Trạng thái:** Proposed method, **chưa được thực nghiệm xác nhận**.  
> **Nguồn nội bộ:** `IDOM_idea_review.pdf` (4 trang; 06/11/2026) và thảo luận về score-level consistency/complementarity qua **K** ODE steps.  
> **Đối tượng sử dụng:** nghiên cứu sinh / kỹ sư triển khai; không phải bản method cuối cho bài báo.  
> **Mục tiêu trước mắt:** kiểm chứng cơ chế cross-modal score feedback có thực sự cải thiện next-item ranking qua các bước ODE hay chỉ tương đương ensemble / iterative fusion thông thường.

---

## 0. Executive decision — đọc trước khi code

**Câu hỏi nghiên cứu:**

> Can an intent-conditioned, score-mediated coupling between modality-specific trajectories exploit complementary evidence **across generation steps** better than independent modality predictors, final-score fusion, and equally deep non-ODE refinement?

**Nguyên tắc thiết kế:**

1. Mỗi modality (ID, text, image) **giữ trajectory và embedding space riêng**. Không ép `x_ID ≈ x_text ≈ x_image`.
2. **Item identity + item-level score** là cầu nối chính giữa các không gian biểu diễn; latent **user intent** chỉ quyết định *khi nào, từ modality nào, và với cường độ nào* tiếp nhận tín hiệu.
3. Score từ step `k` của modality nguồn được chuyển thành **feedback trong embedding space của modality nhận**, rồi đi vào vector field của step `k+1`.
4. Trao đổi phải **synchronous**: tính tất cả modality scores ở step `k` trước khi cập nhật tất cả states sang step `k+1`.
5. **K là độ phân giải solver**, không cố định là 3. Thử nghiệm `K ∈ {1,2,3,5,10}`; `30` chỉ dùng cho kiểm tra bổ sung. Không bắt buộc accuracy tăng đơn điệu theo K.
6. **Không teacher-force target item vào score feedback khi training**. Feedback dùng trạng thái do model tự rollout, tương tự inference; flow-matching supervision từ cặp `(z0, x1)` được tính trong nhánh phụ tách biệt.
7. **Chưa thêm consistency KL giữa các modality**, chưa thêm shared endpoint/intent anchor, và chưa thêm nhiều loss mới trong bản MVP. Trước hết xác định feedback có giá trị độc lập hay không.
8. **Go/no-go:** phải vượt được *separate-model ensemble* và *iterative score refinement không ODE* trong cùng budget/điều kiện; nếu không, không nên claim ODE là contribution.

**Giới hạn các claim:** `intent`, `complementarity`, `score-coupled ODE` đều đang là **giả thuyết cần kiểm chứng**. Gating trên score disagreement **không đồng nghĩa** với đo hoặc bảo toàn unique information theo nghĩa information theory.

---

## 1. Phân biệt bằng chứng đã có và đề xuất mới

### 1.1. Đã có trong báo cáo IDOM (không phải kết quả của method mới)

Dữ liệu: Amazon Beauty theo protocol FMRec/S3Rec, **22,363 users, 12,101 items**; leave-one-out, full ranking; seeds `{1997, 2024, 42}`; validation mỗi 20 epochs, patience 5. Text encoder: frozen `all-MiniLM-L6-v2`, input title, brand, categories và 64 từ đầu description; item-to-ASIN mapping được đối chiếu với raw Amazon 2014 (theo báo cáo).

| Model (3 seeds trừ khi ghi khác) | HR@10 (%) | NDCG@10 (%) | HR@20 (%) | NDCG@20 (%) |
|---|---:|---:|---:|---:|
| FMRec reproduced (v3) | 8.36 ± 0.06 | 4.97 ± 0.02 | 11.71 ± 0.03 | 5.81 ± 0.03 |
| IDOM ID branch | 8.41 ± 0.07 | 4.93 ± 0.07 | 11.82 ± 0.10 | 5.79 ± 0.08 |
| IDOM text branch | 7.02 ± 0.06 | 3.53 ± 0.04 | 10.93 ± 0.10 | 4.51 ± 0.05 |
| **IDOM ID + text fusion, no anchor** | **9.18 ± 0.10** | **5.21 ± 0.07** | **13.11 ± 0.12** | **6.20 ± 0.08** |

Nguồn: báo cáo IDOM, trang 3. **Không có kết quả cho image, intent anchor hoặc score feedback.** Báo cáo ghi nhận `text-only` train riêng có validation HR@10/NDCG@10 `9.76/5.23`, còn text branch trong IDOM chỉ `9.19/4.81` trên validation; đây là dấu hiệu khả dĩ của negative transfer (chưa chứng minh nguyên nhân).

**N-step diagnostic:**

| Euler steps | Fused HR@10 (%) | NDCG@10 (%) | Test-pass time |
|---:|---:|---:|---:|
| 1 | 9.17 ± 0.10 | 5.21 ± 0.06 | 4.4 s |
| 2 | 9.18 | 5.21 | 8.0 s |
| 5 | 9.17 | 5.21 | 19.3 s |
| 10 | 9.17 | 5.21 | 37.0 s |
| 30 | 9.18 | 5.21 | 109.1 s |

Nguồn: báo cáo IDOM, trang 3; một số giá trị chỉ ghi trung bình, không có std. Kết luận đúng phạm vi: **state/time dependence của implementation hiện tại rất yếu**, không phải mọi ODE/flow model đều vô ích.

### 1.2. FMRec đang làm gì?

Theo phân tích implementation trong báo cáo (không phải chúng ta đã chạy kiểm chứng mã nguồn độc lập):

- `z0 ~ N(0,I)`; `x1` là embedding target item.
- Straight interpolation: `x_t = (1-t) z0 + t x1`.
- Endpoint predictor: `x1_hat = f_theta(h, x_t, t)`.
- State/time đi vào history qua `LayerNorm(h) + lambda ⊙ (x_t + emb_t)`, với `lambda ~ N(0.001, 0.001^2)` theo báo cáo.
- ODE sampler của reproduction dùng `v_i = x1_hat_i - z0`, nên với Euler uniform step: `x_K = (1/K) * sum_i x1_hat_i`; nếu `x1_hat_i` gần như bất biến theo state/time, tăng K không giúp ranking.

**Cẩn trọng:** công thức nghiệm tối ưu chỉ với MSE của một model dùng đầy đủ đầu vào là `E[x1 | h, x_t, t]`, **không phải** mặc định `E[x1 | h]`; latter là xấp xỉ khi mạng thực tế bỏ qua `x_t,t`. FMRec còn CE và auxiliary decoder loss, nên cũng không nên suy luận toàn bộ optimum chỉ từ MSE.

**Objective hiện tại trong báo cáo:** mỗi modality dùng `MSE(x1_hat,x1) + 0.2 CE + 0.4 auxiliary decoder loss`; tổng trên modalities. Đó là baseline cần giữ để reproduce. **Không mặc định đem objective nguyên trạng áp dụng cho vector field mới mà không kiểm tra sự tương thích.**

### 1.3. Phần đề xuất mới của tài liệu này

**IDOM-SC** = per-modality flow trajectories + step-wise score extraction + intent-conditioned cross-modal feedback + feedback-conditioned vector field. Không có thí nghiệm nào trong báo cáo PDF kiểm chứng thiết kế này.

---

## 2. Hypotheses có thể bị bác bỏ

- **H1 / Complementary rescue:** score feedback có thể tăng tỷ lệ *ID-wrong → fused-correct* mà không gây nhiều *ID-correct → fused-wrong* hơn ensemble.
- **H2 / Cross-step interaction:** sửa đổi score của modality `n` ở step `k` có ảnh hưởng có đo được tới state **và** ranking của modality `m ≠ n` ở `k+1`.
- **H3 / ODE advantage:** với số tham số, số vòng lặp và FLOPs/latency tương đương, coupled ODE tốt hơn `K`-round non-ODE score refinement.
- **H4 / Intent dependence:** khi cố định user history và candidate information, intent-conditioned gating tốt hơn constant gates / time-only gates / random intent; còn có sự khác nhau có ý nghĩa giữa users, contexts.
- **H5 / Generality:** gain giữ được qua nhiều seeds, datasets, và khi thêm image; không chỉ Beauty ID+Text.

**Giả thuyết phải tách:** Một mô hình cải thiện HR@10 chưa đủ để kết luận H2 hoặc H3. Một branch có khác score branch còn lại cũng chưa đủ để kết luận H1.

---

## 3. Problem setup và notation

- Users `u ∈ U`, items `i ∈ I`, số item `V = |I|`.
- History chỉ gồm sự kiện trước target: `H_u = (i_1,...,i_L)`; target `y_u = i_(L+1)`.
- Modalities `m ∈ M = {ID, T, V}` (giai đoạn đầu `{ID,T}`).
- Item table modality `m`: `E^m ∈ R^(V×d_m)`, hàng `e_i^m`; text/image features có thể frozen rồi qua learned projection. Cùng một **item ID index** xuyên suốt mọi bảng embedding. **Nếu dùng đúng một shared vector-field backbone như IDOM, các input/output của backbone phải có cùng latent dimension `d`; khi `d_m` khác nhau cần modality-specific input/output projections vào/ra `d`.** Trong MVP nên chọn `d_m=d` để giữ implementation đơn giản; `d_m` chỉ biểu thị phiên bản tổng quát.
- History representation `h_u^m ∈ R^(d_h)`; chia sẻ backbone + modality token như IDOM, hoặc tách branch ở ablation.
- **Latent intent** `q_u = IntentEncoder(H_u)`. MVP: `q_u = LN(MLP(h_u^ID))`, chỉ lấy từ lịch sử; có thể mở rộng sang pooled multimodal history ở ablation.
- State `x_k^m ∈ R^(B×d_m)`; time `t_k = k/K`, `k=0,...,K-1`; `Δt = 1/K`.
- Endpoint prediction `xhat_(1,k)^m`; score `s_k^m ∈ R^(B×V)`; probabilities `p_k^m` là phân phối chuẩn hóa sau calibration.
- Full-ranking evaluation: lịch sử đã xem được mask theo protocol; **không mask target**, không dùng target để tạo candidates/gates lúc inference.

### 3.1. Hai lựa chọn vector-field parameterization (không trộn lẫn)

**A — Endpoint predictor (thay đổi ít hơn so với FMRec):**

```math
\hat{x}_{1,k}^m = f_\theta(h_u^m,x_k^m,t_k,c_k^m,e_m),
\qquad
v_k^m = \frac{\hat{x}_{1,k}^m-x_k^m}{1-t_k},
\qquad
x_{k+1}^m = x_k^m + \Delta t\, v_k^m.
```

- Chỉ evaluate tại `t_k < 1`, không chia cho zero. Với uniform Euler `t_(K-1)=(K-1)/K`, denominator nhỏ nhất `1/K`.
- Để score ở step `k`, dùng `xhat_(1,k)^m` (không dùng noisy `x_k^m`).
- Feedback `c_k^m` nên đi vào **predictor/velocity computation**, không chỉ cộng vào final score, để có cross-step dependency.
- Nếu `f` không nhạy với state, số step vẫn không có giá trị; cần test bằng perturbation.

**B — Direct velocity predictor (sạch hơn về mặt flow matching):**

```math
v_k^m = v_\theta(h_u^m,x_k^m,t_k,c_k^m,e_m),
\qquad
\hat{x}_{1,k}^m = x_k^m+(1-t_k)v_k^m,
\qquad
x_{k+1}^m = x_k^m+\Delta t\,v_k^m.
```

- Teacher flow supervision `v_true^m = x1^m - z0^m` trên `x_t^m=(1-t)z0^m+t x1^m`.
- Đây là phương án dễ giải thích hơn nếu muốn claim genuine flow dynamics, nhưng **khác FMRec reproduction**. Chỉ thử sau khi MVP endpoint predictor cho tín hiệu tích cực, hoặc đặt nó thành một ablation độc lập.
- Không áp dụng một cách máy móc các auxiliary losses của FMRec lên velocity khi chưa xác định mục tiêu và decoder đúng.

**Khuyến nghị triển khai:** Pha đầu giữ **A** để kiểm tra ý tưởng feedback với thay đổi nhỏ nhất; Pha xác nhận ODE sử dụng thêm **B**. Mỗi biến thể train từ đầu và log riêng.

---

## 4. Score coupling: modality khác gửi gì, modality nhận nhận gì?

### 4.1. Extract per-modality item scores

Ở mỗi step:

```math
s_k^m(i)= \langle \hat{x}_{1,k}^m,e_i^m \rangle,
\qquad
p_k^m(i)=\operatorname{softmax}_i(s_k^m(i)/\tau_m),
```

`τ_m>0` điều chỉnh calibration (học được hoặc tune trên validation; nếu learnable phải regularize và báo cáo). Softmax dùng cùng **candidate universe**/mask across modalities. Với `V=12,101`, giai đoạn đầu ưu tiên **full-item scores** để tránh bias do candidate truncation.

Lưu ý: `sum_m w_m * log_softmax(s_m)` có **cùng ranking** với `sum_m w_m*s_m` khi `w_m` chỉ phụ thuộc user (không phụ thuộc candidate), vì log normalizer không phụ thuộc item. Nó **không tự scale-calibrate** modalities. Cần baseline tune `w_m,τ_m` bằng validation.

### 4.2. Score discrepancy chỉ là tín hiệu đề xuất, chưa phải complementarity

Một bản MVP không cần attention matrix `V×V`:

```math
d_{n\to m,k}(i)=[p_k^n(i)-p_k^m(i)]_+,
\quad n\neq m.
```

- Nếu source `n` thích item `i` hơn receiver `m`, `d>0`.
- Nếu chúng đồng ý, `d` nhỏ.
- **Nhưng source có thể sai**; không dùng `d` làm supervision về “unique information”.
- Test đối chứng signed discrepancy `(p_n-p_m)` và agreement-only feedback; xem §10.

### 4.3. Intent-dependent reliability gate

Trước tiên dùng **gate theo user × source/receiver × step** để tránh thêm gate item-specific quá sớm:

```math
\alpha_{n\to m,k}(u)
=\operatorname{sigmoid}\!
\left(g_\phi(q_u,\operatorname{summary}(p_k^n),
\operatorname{summary}(p_k^m),t_k,\mathrm{emb}(n),\mathrm{emb}(m))\right).
```

- Summary đơn giản: normalized entropy, top-1/top-2 margin, max probability, mean/variance logits; **không dùng y**.
- Chỉ từ iteration 2 trở đi, mới cân nhắc gate item-specific `α(u,i,n,m,k)` nếu user-level gate chưa đủ.
- Có thể dùng normalized temperature + layer norm để giải quyết dynamic range lớn giữa embedding tables.

### 4.4. Score-to-state projection qua item identity

Thông tin source `n` gửi đến target space `m`:

```math
w_{n\to m,k}(i)=\alpha_{n\to m,k}(u)\,[p_k^n(i)-p_k^m(i)]_+,
```

```math
r_{n\to m,k}=
\frac{\sum_{i\in\mathcal I} w_{n\to m,k}(i)e_i^m}
{\epsilon+\sum_{i\in\mathcal I}w_{n\to m,k}(i)}.
```

Khi tổng weights bằng 0, định nghĩa feedback bằng zero. **Điểm mới so với direct feature alignment:** text gửi *item preference weights*, nhưng receiver ID/visual tự dùng `E^ID/E^V` để biểu diễn những item đó. Không cần project text embedding thành ID embedding.

Tổng feedback nhận bởi modality `m`:

```math
r_{m,k}=\sum_{n\ne m}r_{n\to m,k},
\qquad
c_{m,k}=G_{\psi,m}(\operatorname{LN}(r_{m,k}),q_u,t_k,x_k^m).
```

`G` có output dimension `d_m`. Dùng **zero-initialized last layer**, residual strength `β` nhỏ để lúc khởi tạo mô hình gần baseline, tránh phá hỏng pretrained branch.

### 4.5. Coupled vector field

**Cách đơn giản trong MVP:**

```math
\tilde v_k^m = v_{\theta,m}(x_k^m,t_k,h_u^m)
+\beta(t_k)G_{\psi,m}(r_{m,k},q_u,x_k^m,t_k),
```

```math
x_{k+1}^m = x_k^m + \Delta t\,\tilde v_k^m.
```

Đây là **velocity guidance**, không phải forced endpoint alignment. Nếu muốn dùng endpoint parameterization A, có thể làm equivalent implementation bằng cách cập nhật `x_k` với `v=(xhat1-x_k)/(1-t_k)` cộng `βG`; score được recompute từ state mới. Đảm bảo `G` không bị bỏ qua bởi lớp transformer hoặc output scoring head.

Đề xuất `β(t)` để ablate, chưa khẳng định tối ưu:

- Constant small: `β(t)=β0` (**default**).
- Late guidance: `β(t)=β0*t`.
- Middle guidance: `β(t)=β0*sin(πt)`.

Chọn bằng validation; có `β=0` làm **no-feedback control**. Theo dõi gradient norm và tỷ lệ `||βG||/||v_base||` để tránh feedback áp đảo flow.

### 4.6. Synchronous step semantics

**Đúng:** ở step `k`, tính `s_k^ID,s_k^T,s_k^V` từ cùng bộ states hiện tại; sau đó tính **tất cả** messages `r_{n→m,k}`; cuối cùng cùng lúc cập nhật states.

**Sai:** update ID xong dùng score ID mới để update Text ngay trong step đó, nhưng Image lại nhìn ID/Text ở hai phiên bản thời gian khác nhau. Nếu dùng asynchronous schedule thì phải định nghĩa riêng và làm ablation.

### 4.7. Cross-step và backward gradient

- Inference forward: `s_k^T -> r_(T→ID,k) -> x_(k+1)^ID -> s_(k+1)^ID`.
- Training backward: `L_final -> x_K^ID -> r_(T→ID,K-1) -> s_(K-1)^T`, **nếu** không stop-gradient ở đường feedback và không detach trajectory.
- Chạy control `detach(messages)` để kiểm tra lợi ích của end-to-end backprop qua modality.
- `topk` index selection không khả vi; giá trị score trên candidate set vẫn có gradient. Tốt nhất **khởi đầu không dùng top-L** để tránh khó giải thích.

---

## 5. Training protocol — phần dễ mắc lỗi nhất

### 5.1. Hai nhánh objective phải được tách

**Nhánh A: Self-generated rollout / ranking loss**

1. Encode history `H_u`; tạo `q_u` chỉ từ `H_u`.
2. Sample `z0^m ~ N(0,I)` cho từng modality (mặc định independent; thử shared-noise ở ablation).
3. Khởi tạo `x_0^m=z0^m`; **không** tạo `x_k=(1-t_k)z0+t_k*x1` để đưa vào score-feedback.
4. Unroll K steps; tính `score -> message -> state update` như inference.
5. Tính final fused scores và `CE(y, score_final)`; optionally thêm branch CE và intermediate CE với weights nhỏ.

**Nhánh B: Teacher flow-matching supervision**

Tách khỏi feedback tự rollout để duy trì khả năng học vector field:

```math
x_t^m=(1-t)z_0^m+t e_y^m,
\qquad t\sim U(0,1).
```

- Endpoint option A: `L_FM^m = ||f_theta(h_u^m,x_t^m,t, c=0) - e_y^m||²`; có thể giữ auxiliary CE/decoder từ FMRec nếu đã chứng minh chúng hữu ích, và giữ hệ số trong bảng ablation.
- Velocity option B: `L_FM^m = ||v_theta(h_u^m,x_t^m,t,c=0) - (e_y^m-z0^m)||²`.
- Có thể sau đó nghiên cứu teacher supervision cho guided field, nhưng **không** tạo guidance teacher-forced từ ground-truth noisy trajectory để giả làm inference guidance.

**Vì sao cần tách?** Khi training bằng `x_t` chứa `e_y`, scores tại intermediate steps có thể leak thông tin target vào cross-modal feedback; inference bắt đầu từ `z0` không có target. Training CE từ chính các teacher-forced scores có thể tạo gains ảo.

### 5.2. Loss đề xuất để bắt đầu

```math
L= L_{\mathrm{rec,final}}
+\lambda_{\mathrm{flow}}\sum_m L_{\mathrm{FM}}^m
+\lambda_{\mathrm{branch}}\sum_m \mathrm{CE}(s_{\mathrm{final}}^m,y)
+\lambda_{\mathrm{inter}} L_{\mathrm{intermediate}}.
```

- `L_rec,final = CE(s_fused,K, y)` trên full item universe, với lịch sử masked giống FMRec.
- **MVP:** `lambda_inter = 0`; **không có** `KL(p_ID||p_T)`, `MSE(x_ID,x_T)`, contrastive cross-modal endpoint alignment.
- `lambda_flow` và `lambda_branch`: tune trên validation, bắt đầu từ scale làm branch hoạt động ổn như reproduction. **Không tự nhận các coefficient trong ví dụ là tối ưu.**
- Nếu dùng original FMRec auxiliary decoder: báo cáo riêng `+L_aux` và `-L_aux`, không double-count supervision.
- `s_fused,K` lấy từ **endpoint predictions tại step cuối hoặc final state** theo một định nghĩa cố định xuyên suốt train/val/test. Không trộn hai cách tính khi so baselines.

### 5.3. Fused ranking để train/evaluate

Start with transparent calibration/fusion:

```math
s_{\mathrm{fused},K}(i)=\sum_m \pi_m(q_u)\,s_K^m(i)/\tau_m,
\qquad \pi_m\ge0,\quad\sum_m\pi_m=1.
```

Ablate: `(1) fixed equal`, `(2) validation-tuned constant`, `(3) history-conditioned gate`, `(4) item-specific gate` only if needed. Không đồng thời mở rộng gate complexity và ODE complexity ở thí nghiệm đầu.

**Khuyến nghị:** để chứng minh riêng tác dụng coupling, ban đầu cố định một phương thức fusion và giữ nguyên ở `feedback on/off`.

### 5.4. K thay đổi theo training/inference

- Train với `K_train=3` hoặc sample `K_train∈{2,3,5}` (hai lựa chọn, **phải báo cáo đúng**).
- Đánh giá `K_test∈{1,2,3,5,10}`. `K_test=1` trong scheme synchronous score feedback chỉ có thể sử dụng guidance trong chính bước đầu nếu được định nghĩa rõ; với thiết kế *message tác động bước tiếp theo*, `K=1` **không tạo vòng cross-step refinement**. Nó là baseline kiểm tra một-step.
- Nếu train ở K cố định, test K khác là **solver generalization** chứ chưa phải bất biến về số bước; cần report rõ.
- Không dùng learned `step_id_embedding` riêng cho mỗi K; dùng continuous `t`.
- Ghi chính xác `t_k`, discretization, numerics, beta schedule, `epsilon`; không đổi solver tại inference mà không retrain/ablate.

---

## 6. Pseudocode (implementation blueprint)

### 6.1. Định nghĩa shapes

| Tensor | Shape | Ghi chú |
|---|---|---|
| `hist` | `[B,L]` | item IDs trong history |
| `target` | `[B]` | next item ID, **chỉ dùng cho loss** |
| `E[m]` | `[V,d_m]` | cùng item indexing |
| `h[m]` | `[B,d_h]` | modality-specific history signal |
| `q` | `[B,d_q]` | intent từ observed history |
| `x[m]` | `[B,d_m]` | ODE state |
| `xhat[m]` | `[B,d_m]` | predicted endpoint |
| `scores[m]` | `[B,V]` | item scores |
| `probs[m]` | `[B,V]` | calibrated distribution |
| `msg[m]` | `[B,d_m]` | feedback trong receiver embedding space |

### 6.2. PyTorch-like high-level pseudocode

```python
# Algorithm 1: synchronous score-coupled rollout
# item_tables[m] shares the same item ordering for all modalities.

def rollout(history, item_tables, K, temperature, beta):
    h = {m: history_encoder(history, m) for m in modalities}
    q = intent_encoder(h['id'])  # no target exposure
    x = {m: torch.randn(history.size(0), d[m], device=device)
         for m in modalities}

    for k in range(K):
        t = k / K
        dt = 1.0 / K

        # A. Score ALL modalities using their states at the SAME step.
        # These predictions cannot be teacher-forced by target embeddings.
        endpoint = {}
        score = {}
        for m in modalities:
            endpoint[m] = endpoint_predictor(m, h[m], x[m], t)
            score[m] = endpoint[m] @ item_tables[m].T   # [B,V]
            score[m] = mask_history_items(score[m], history)

        prob = {m: torch.softmax(score[m] / temperature[m], dim=-1)
                for m in modalities}

        # B. Compute ALL cross-modal messages using only step-k scores.
        message = {}
        for receiver in modalities:
            terms = []
            for source in modalities:
                if source == receiver:
                    continue
                diff = (prob[source] - prob[receiver]).clamp_min(0)
                gate = reliability_gate(q, prob[source], prob[receiver],
                                        t, source, receiver)  # [B,1]
                w = gate * diff                          # [B,V]
                denom = w.sum(dim=1, keepdim=True)       # [B,1]
                # When no positive discrepancy, message should be zero.
                term = (w @ item_tables[receiver]) / (denom + 1e-8)
                terms.append(term)
            message[receiver] = sum(terms)  # [B,d_receiver]

        # C. Simultaneous state updates; don't mutate x inside source loop.
        next_x = {}
        for m in modalities:
            base_velocity = (endpoint[m] - x[m]) / max(1.0 - t, 1e-6)
            guided_velocity = feedback_adapter(m, message[m], q, x[m], t)
            next_x[m] = x[m] + dt * (
                base_velocity + beta(t) * guided_velocity
            )
        x = next_x

    # OPTION: final ranking must have a single consistent definition.
    # Recompute predicted endpoints at final t<1, or decode final x state.
    # Choose one and use it identically in training and all comparisons.
    final_score = decode_final_scores(x, h, item_tables, K)
    fused_score = score_fusion(final_score, q)
    return fused_score, final_score
```

**Engineering detail:** `decode_final_scores` must not call endpoint predictor at `t=1` if the parameterization assumes `(1-t)>0`; easiest is to define final scores as `x[K] @ E[m].T`, then use exactly that definition in all runs. Conversely, for the unmodified FMRec comparison where final scores historically came from endpoint prediction, keep its original convention. If comparing components, explicitly test equivalence/cost of the two decoders.

**Potential issue:** khi `β=0`, `endpoint` trong loop không thấy messages, và decoder cuối có thể phụ thuộc rất yếu vào state. Đây là expected negative control; test gradients/sensitivity.

### 6.3. Minimal training loop

```python
for hist, y in loader:
    K = sample_train_steps()  # e.g. fixed 3, or sampled from {2,3,5}
    s_fused, s_mod = rollout(hist, E, K, tau, beta)
    loss_rank = F.cross_entropy(s_fused, y)
    loss_branch = sum(F.cross_entropy(s_mod[m], y) for m in modalities)
    loss_flow = teacher_flow_loss_from_separate_xt(hist, y, E)
    loss = loss_rank + lam_branch*loss_branch + lam_flow*loss_flow
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
```

**Chú ý memory/compute:** backward qua K steps và score `[B,V]` cho từng modality có thể chiếm VRAM. Trước tiên dùng full ranking trên Beauty; sau đó mới cân nhắc checkpointing/gradient accumulation/union top-L. Gradient checkpointing không thay đổi semantics; top-L có thể thay đổi semantics đáng kể.

---

## 7. Thiết kế baseline bắt buộc

### 7.1. P0 — feasibility và giá trị ngoài ensembling

| ID | Baseline/control | Cần trả lời |
|---|---|---|
| B0 | FMRec reproduction ID-only (`K=1,30`) | kiểm tra reproducing |
| B1 | Text-only model train riêng | strength của content branch |
| B2 | **FMRec + Text-only ensemble**, tune weights/temperature trên validation | multimodal gain có vượt ensemble không? |
| B3 | IDOM original shared field, final fusion (`K=1,30`) | baseline hiện tại |
| B4 | Direct one-step predictor, không dùng `x_t,t` | ODE có đóng góp gì ngoài direct prediction? |
| B5 | Shared field vs separate modality backbones | parameter sharing có gây negative transfer không? |

**P0 là gate đầu tiên.** Nếu ensemble đã >= IDOM, chưa nên đầu tư thêm phần intent/ODE chỉ để sửa một vấn đề chưa được kiểm chứng.

### 7.2. P1 — isolate cross-modal feedback

| ID | Ablation | Cố định các yếu tố khác |
|---|---|---|
| C0 | Original IDOM no feedback | same encoder / loss / fusion |
| C1 | Feedback **at final score only** | không có state update |
| C2 | **K-round iterative score refinement, không ODE** | same K, comparable params/FLOPs |
| C3 | K-step **ODE, no cross-modal feedback** | isolate contribution of steps/state |
| C4 | **K-step coupled ODE score feedback** | proposed |
| C5 | C4 with `β=0` | no guidance control |
| C6 | C4 with `q` replaced by constant/zero | role of intent |
| C7 | C4 with random/shuffled intent across users | kiểm tra user-specific relevance |
| C8 | C4 with one-way `Text→ID` vs `ID→Text` | directionality / negative transfer |
| C9 | C4 with `detach(message)` | value of cross-modal gradient path |
| C10 | C4 with score-source shuffled among users | phá correspondence nhưng giữ distribution |

**Control C2 rất quan trọng:** nếu coupled ODE không thắng iterative MLP/refinement dùng cùng số vòng K, không thể claim ODE là nguyên nhân của gain.

### 7.3. P2 — isolate message design và solver

- Positive discrepancy `[p_n-p_m]_+` vs signed `(p_n-p_m)` vs top-k-only discrepancy.
- Source-conditioned gate vs constant gate; user-level gate vs item-specific gate.
- Shared receiver item table (proposed) vs learned projection of source representation; score-mediated cross-space vs direct embedding alignment.
- Endpoint predictor vs direct velocity FM target; state/time injection `lambda ∈ {0,0.1,1.0}` **chỉ theo đúng implementation report** hoặc riêng AdaLN/FiLM; tránh đưa parameterization vào test mà không train lại.
- `K_train` fixed vs random, `K_test∈{1,2,3,5,10,30}`; `β(t)` constant / increasing / middle schedule.
- Full-item messages vs shared **union top-L** candidates; `L ∈ {50,100,200}` sau khi full-item có kết quả.
- Frozen text features vs learned projection; test thêm image khi available.
- Same `z0` vs independent `z0` across modalities; multiple `z0` samples tại test để kiểm tra variance/stability.

---

## 8. Candidate set, efficiency, và những chỗ dễ tạo improvement ảo

### 8.1. Full-item trước, top-L sau

Trong giai đoạn Beauty `V=12,101`, bắt đầu từ `[B,V]` full scores để dễ chứng minh end-to-end feedback. Truyền score qua item ID rồi nhân `w @ E_receiver`; **không cần** ma trận tương tác item–item kích thước `V×V`.

Nếu phải dùng `C_k = union_m topL(s_k^m)`:

- `C_k` phải được tạo từ **current model scores**, không thêm ground-truth item bằng tay vào candidate set lúc training.
- Cùng `C_k` cho mọi modality trong một step; bảo đảm có mapping item indices nhất quán.
- Với item ngoài `C_k`, định nghĩa cách xử lý feedback (zero) và vẫn tính full-ranking **final scores** nếu protocol là full ranking.
- Vì top-L selection rời rạc và có thể thay đổi theo step, hệ thống là piecewise/non-smooth; **không tự động claim globally smooth ODE vector field**.
- Khi source-only item không lọt top-L chung, không thể rescue; báo cáo **candidate recall / missed-target rate** cùng accuracy.

### 8.2. Computational comparison

Báo cáo gốc: IDOM ID+text khoảng 62.3 s/epoch, 4.9 GB peak GPU, 4.8h mean total training so với FMRec 34.2 s/epoch, 2.6GB, 1.9h; khi `K=1`, val pass IDOM 4.1 s vs FMRec 1.2 s (benchmarks khác test pass; không cộng/trộn số liệu).

Với score-feedback mới, phải log:

- Epoch time, validation/test per-pass time, p50/p95 per-user or per-batch latency (same hardware/batch).
- Peak VRAM, params, FLOPs (ước lượng cùng conventions), số encoder/ODE forwards.
- Score computation `[B,V]`, feedback `w@E`, extra adapter và gating overhead.
- Latency vs accuracy plot theo K. **Không claim IDOM nhanh hơn FMRec** chỉ vì cả ba modality chạy trong một shared network.

### 8.3. Training/inference leakage checklist

- [ ] Không dùng target item để tạo query `q`, gate, feedback, candidate pool ở inference.
- [ ] Free-running rollout từ `z0` trong ranking loss; không dùng `x_t=(1-t)z0+t e_y` để tính intermediate score-feedback.
- [ ] `L_FM` teacher-forced nằm nhánh phụ, không làm rò target vào conditional score path.
- [ ] Item metadata có thể có cho test items nếu protocol cho phép; không dùng **test interactions/reviews của user** để xây content hoặc ranking history.
- [ ] Same item-ASIN-text-image mapping across folds; không có missing/duplicate item IDs.
- [ ] Exclude previously consumed items exactly as baseline protocol; không mask positive target.
- [ ] Tune gate, temperature, fusion, K on **validation only**; test once per locked configuration.
- [ ] Random seed kiểm soát weight init / order / z0; report randomness sensitivity.

---

## 9. Metrics và diagnostics (không chỉ HR/NDCG)

### 9.1. Main accuracy

- Full-ranking `HR@{5,10,20}`, `NDCG@{5,10,20}`, optional MRR. Giữ `HR@10`, `NDCG@10` là chính để đối chiếu FMRec.
- `mean ± std` trên **cùng seeds** cho tất cả baselines; tối thiểu 3 seeds exploratory, tăng seeds/paired user-level bootstrap khi kết luận.
- Paired comparisons theo cùng user/test targets; bootstrap 95% CI cho delta, permutation test khi phù hợp; multiple-comparison correction (ví dụ Holm) cho các comparisons chính. Không chỉ nhìn overlap giữa std của seeds.
- Group analyses: short vs long history; popular vs long-tail items; text-rich vs text-poor; về sau image quality / missing modality.

### 9.2. Complementarity / rescue-harm table

Từ two branches cùng checkpoint, đo theo từng user với một `@10` cố định:

| Nhóm | ID đúng? | Text đúng? | Ý nghĩa |
|---|---|---|---|
| A | Có | Có | agreement, dễ |
| B | Có | Không | cần tránh phá ID |
| C | Không | Có | **oracle rescue opportunity** |
| D | Không | Không | khó: cần combined/synergistic evidence hoặc capacity |

Với fusion/coupling, tính:

```text
rescue_rate = #(ID wrong, final correct) / #(ID wrong)
harm_rate   = #(ID correct, final wrong) / #(ID correct)
net_rescue  = #(ID wrong, final correct) - #(ID correct, final wrong)
```

Riêng nhóm C, báo cáo thêm `rescue_C = #final correct among C / #C`. Nhóm D có thể cải thiện nhưng **không tự động chứng minh synergy**. Bảng này phải tính trên same users, same candidate universe và same mask.

### 9.3. Cross-step causal intervention / diagnostics

**Can thiệp có kiểm soát:** giữ nguyên `H_u,z0,model`, thay `s_k^T` bằng score shuffled/zero tại một step, đo tác động lên `s_(k+1)^ID` và final ranking. Thao tác can thiệp trên computational graph **để chứng minh đường truyền có hoạt động**, không phải claim quan hệ nhân quả ngoài mô hình.

Log ít nhất:

1. `Δ_score_ID(k→k+1) = ||s_(k+1)^ID-s_k^ID||` và tương tự T/V.
2. Score rank correlation (Spearman/Top-10 overlap) giữa các bước, và giữa modalities.
3. `||message_(n→m,k)||`, `||βG||/||v_base||`, gate values entropy/variance.
4. `Top-k target rank` qua từng step (dùng target chỉ để **evaluate**).
5. Fraction target moves into top-10, moves out of top-10, split by B/C groups above.
6. State sensitivity to perturbing `x_k`, `t_k` while holding `H` fixed; effect of changing `z0` at inference.
7. Detect collapse: similarity between modality score distributions **before vs after coupling**, branch HR/NDCG và score entropy. Score agreement tăng mà branch performance giảm có thể là over-alignment.
8. `K=1` vs `K>1` accuracy/cost, and `β=0` vs `β>0`.

**Minimum mechanistic evidence:** with same `H,z0`, disabling `T→ID` must measurably affect ID score at later steps; the change should associate with higher correct rescues (not merely more disagreement).

---

## 10. Experiment roadmap và decision gates

### Stage A — Reproduce và freeze protocol

- [ ] Xác nhận FMRec 3 seeds; cùng preprocess, seed, early stopping và ranking evaluation như báo cáo.
- [ ] Hoàn tất separately trained text-only và fusion baseline, tune weights/temperatures trên validation.
- [ ] Test direct one-step predictor `f(h)` và FMRec `λ=0`/state sensitivity.
- [ ] Audit item mapping, test leakage, calibration và compute.

**Exit:** có bảng B0–B5 đáng tin. **Nếu riêng ensemble > IDOM hiện tại, coi là tín hiệu cần chứng minh thêm trước khi gọi shared-field là advantage.**

### Stage B — Minimal coupled score feedback (ID+Text; Beauty)

- [ ] Giữ architecture và data hiện tại; tạm dùng `K_train=3`, evaluate `K_test={1,2,3,5,10}`.
- [ ] Thêm per-step calibrated full-item scores.
- [ ] Message bằng positive discrepancy + user-level gate + receiver item table.
- [ ] Add zero-init receiver adapter to velocity; sync updates.
- [ ] Train **free-rollout CE** và nhánh FM riêng; không KL modality.
- [ ] Chạy controls `β=0`, `q=constant`, `Text→ID only`, `ID→Text only`, stopgrad, shuffled scores.
- [ ] Log rescue/harm, cross-step intervention, state dependence, runtime.

**Exit:** coupling vượt fixed/tuned ensemble và cải thiện net rescue, không collapse branch. Nếu không đạt, dừng hướng ODE-coupling hoặc đơn giản hóa về fusion/iterative predictor.

### Stage C — Test ODE-specific contribution

- [ ] Non-ODE K-round MLP iterative refinement, cùng số iterations, gần cùng params/FLOPs.
- [ ] ODE `K` steps `β=0` / `β>0`, plus direct-velocity variant.
- [ ] K generalization (train K fixed vs random), noise sensitivity, solver/time discretization.
- [ ] Mechanistic proof step-to-step corrections actually improve ranking.

**Exit:** coupled ODE phải có lợi ích **độc lập** so với iterative refinement non-ODE, nếu paper muốn claim ODE. Nếu không, chuyển framing thành *iterative score-mediated multimodal preference refinement*.

### Stage D — Multimodal validation / publication-grade

- [ ] Thêm **image** trên datasets có raw image features/mapping đáng tin; modality missingness handling.
- [ ] Nhiều datasets (ngoài Beauty; ví dụ Sports/Toys nếu đúng protocol và đầy đủ features).
- [ ] Strong baselines: generative/flow models và multimodal SR models, same preprocessing/eval.
- [ ] Paired significance, 5+ seeds nếu nguồn lực cho phép, ablations, complexity, gate visualizations.
- [ ] Novelty check lại ngay trước framing cuối; không dùng claim “parallel multimodal flows chưa ai làm”.

**Không bắt buộc mọi experiment ngay:** ưu tiên Stage A, B; C và D chỉ chạy khi B có tín hiệu rõ.

---

## 11. Failure modes, cách chẩn đoán và điều chỉnh tối thiểu

| Triệu chứng | Giả thuyết nguyên nhân | Kiểm tra trước khi thêm module |
|---|---|---|
| `K>1 ≈ K=1` | state/time vẫn bị ignore; feedback không thay prediction | perturb `x`, `t`, `β=0`, check `||G||`, gradient path |
| Có score changes nhưng HR không tăng | feedback khuếch đại modality sai | rescue/harm, gate entropy, label-free reliability; source direction ablation |
| Text branch yếu đi khi joint training | shared field ưu tiên ID/negative transfer | separate backbones, branch auxiliary loss, per-modality adapters |
| Model học `p_ID≈p_Text` và giảm unique rescue | quá alignment/homogenization | bỏ KL/alignment, chỉnh branch loss, bớt feedback strength |
| Gain chỉ khi training teacher-forced | leakage/train–inference mismatch | free rollout từ z0; input audit; identical inference path |
| Gain chỉ với top-L chứa positive | oracle candidate leakage | no target injection, full-item baseline, candidate recall |
| Improvement nhưng latency quá cao | K times full scoring, backprop | accuracy-cost frontier, candidate approximation sau baseline, K nhỏ nếu đủ |
| `z0` thay đổi khiến ranking rất bất ổn | stochastic endpoints không được điều kiện hóa tốt | eval multiple noise seeds, standardized noise, state/velocity regularization thử sau |
| Non-ODE iterative baseline bằng hoặc hơn | benefit là iterative fusion, không phải flow | đổi framing; đừng tăng K vô nghĩa |
| Gate output gần constant | intent chưa cung cấp signal | zero/shuffle-intent ablation, analyze per-user gate variance |

---

## 12. Reporting template cho mỗi run

**Config:**

```yaml
experiment_id: "idom_sc_beauty_idtext_v1"
dataset: "Amazon Beauty (FMRec protocol)"
modalities: [id, text]
seed: 42
history_length: "same as FMRec config"
backbone: "FMRec-compatible shared field"
parameterization: "endpoint"  # or direct_velocity
K_train: 3
K_test: [1, 2, 3, 5, 10]
item_scoring: "full-ranking"
score_temperature: {id: 1.0, text: 1.0}  # placeholders; tune on val
fusion: "validation-tuned-constant"
feedback:
  enabled: true
  discrepancy: "positive_probability_difference"
  gate: "user_intent_pair_step"
  mapping: "weighted_receiver_item_embeddings"
  update: "synchronous_velocity_guidance"
  beta_schedule: "constant"
  beta0: null  # choose on validation
  detach_message: false
loss:
  final_rec_ce: true
  separate_teacher_flow: true
  branch_ce: true
  intermediate_ce: false
  cross_modal_KL: false
sampling:
  z0: "independent_normal_per_modality"
evaluation:
  protocol: "leave-one-out, full-ranking"
  mask_seen_items: true
  metrics: [HR10, NDCG10, HR20, NDCG20]
compute:
  gpu: "record exact GPU name"
  batch_size: null
  peak_vram_gb: null
  train_seconds_per_epoch: null
  val_seconds_per_pass: null
results:
  HR10: null
  NDCG10: null
  rescue_rate: null
  harm_rate: null
  cross_step_intervention_delta: null
```

**Report table cho tất cả methods:**

| Model | K train/test | HR@10 | NDCG@10 | HR@20 | NDCG@20 | rescue | harm | sec/test pass | peak GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 Separate ensemble | –/1 | | | | | | | | |
| B3 Original IDOM | 30/1 | | | | | | | | |
| C2 Iterative non-ODE | K/K | | | | | | | | |
| C3 ODE no feedback | K/K | | | | | | | | |
| C4 Score-coupled ODE | K/K | | | | | | | | |

**Figures tối thiểu nếu có paper:** (1) flow diagram states/scores/messages; (2) gain-vs-K + latency; (3) rescue/harm by disagreement group; (4) message strength/step-wise target rank changes; (5) branch collapse/specialization analysis.

---

## 13. Novelty boundary / related work đã xác minh (06/11/2026)

**Cảnh báo:** một số paper còn là preprint, và các summary dưới đây chỉ nhằm định vị contribution, không thay thế việc đọc method đầy đủ trước khi chốt novelty.

1. **FMRec**, *Flow Matching Based Sequential Recommender Model*, IJCAI 2025. Nền tảng flow matching SR; repo được công bố cùng bài: https://www.ijcai.org/proceedings/2025/346 và https://github.com/FengLiu-1/FMRec . **Overlap:** flow generation từ user history; IDOM-SC khác ở cross-modal coupling, cần chứng minh bằng experiment.
2. **FAVE**, *Flow-based Average Velocity Establishment for Sequential Recommendation*, SIGIR 2026. Prior mismatch, linear redundancy, one-step average velocity, semantic anchor prior: https://doi.org/10.1145/3805712.3809653 ; https://arxiv.org/abs/2604.04427 . **Overlap:** phê bình multi-step inefficiency. Không claim chỉ mình phát hiện “unnecessary steps”.
3. **ConFM**, *Contrastive Flow Matching for Sequential Recommendation*, IJCAI 2026: https://ijcai.org/proceedings/2026/305 . **Overlap:** nghiên cứu averaged flow, user preference discriminability; khác cross-modal score messages.
4. **OT-FlowRec**, *Optimal Transport-Guided Multi-Bridge Flow for Multimodal Sequential Recommendation*, preprint 10/2026: https://arxiv.org/abs/2610.16038 . **Overlap:** parallel flows; họ theo **interest**, IDOM-SC theo **modality**. Không claim parallel flows chưa được làm.
5. **UHIFlow**, *Anatomy of a Decision: Uncertainty-aware Hierarchical Intent Learning via Flow Matching for Multimodal Recommendation*, preprint 08/2026: https://arxiv.org/abs/2609.29609 . **Overlap:** multimodal flow + intent; khác mục tiêu uncertainty / hierarchical intent so với feedback item-ranking qua step.
6. **DMIRec**, *Disentangled implicit and explicit multimodal intent learning for sequential recommendation*, *Multimedia Systems*, 2026: https://doi.org/10.1007/s00530-026-02591-5 . **Overlap:** intent/shared-specific design; intent-conditioned alignment đơn thuần không đủ novelty.
7. **DuAF-MAT**, *Capturing Dynamic User Interests Under Modality Imbalance for Multimodal Sequential Recommendation*, AAAI 2026: https://doi.org/10.1609/aaai.v40i18.38544 . **Overlap:** adaptive modality contributions; generic gating không đủ novelty.
8. **CAMMSR**, *Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation*, ICDE 2026: https://doi.org/10.1109/ICDE65706.2026.00166 . **Overlap:** cross-modal synergy/adaptive fusion; cần isolate gain ngoài fusion.
9. **Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation**, WWW 2025: https://doi.org/10.1145/3696410.3714676 . **Overlap:** adaptive multimodal sequential preference modeling.

**Potential novelty wording (chỉ sau khi experiments xác nhận):**

> *Intent-conditioned score-mediated coupling enables modality-specific flow trajectories to exchange complementary item-level preference evidence across integration steps, improving ranking beyond final-score fusion and matched iterative refinement while avoiding representation collapse.*

**Không dùng các claim sau trước khi có bằng chứng:** "first multimodal flow", "first parallel modality trajectories", "ODE improves because more steps", "intent ensures complementarity", "our method learns unique information".

---

## 14. Checklist giao cho người thực hiện / Definition of done

### Deliverables Stage A

- [ ] Reproduction config + checkpoints/seed log cho FMRec, text-only, IDOM.
- [ ] CSV per-user scores cho ID-only, Text-only, ensemble (validation và test tách riêng).
- [ ] Fixed/tuned fusion weights & temperature (validation); direct predictor control.
- [ ] Report 1 trang: `IDOM vs separate ensemble`, accuracy + speed + 95% paired CI.

### Deliverables Stage B

- [ ] Code `score_coupler.py`: `calibrate -> discrepancy -> gate -> receiver_embedding_weighted_sum`.
- [ ] Code `coupled_solver.py`: K-step synchronous rollout, `β=0` control, checkpoint.
- [ ] Training code tách `free_rollout_rank_loss` và `teacher_flow_loss`.
- [ ] Unit tests: shapes, item ID mapping, zero-feedback, gradient flow, synchronous semantics, no-target-in-feedback.
- [ ] Diagnostic CSV theo user/step/modality: scores/ranks/gates/messages/velocity norms.
- [ ] Experiments C0–C10 (có thể prioritize C0–C7 trước).
- [ ] 2×2 rescue/harm table và 1 ablation summary.

### Deliverables Stage C (nếu B có tín hiệu)

- [ ] Non-ODE matched iterative baseline.
- [ ] `K`/latency Pareto plot.
- [ ] Direct-velocity version & state sensitivity diagnostic.
- [ ] Decision memo: **keep ODE** / **reframe as iterative score refinement** / **stop**.

**Stop conditions:** Không thắng ensemble đáng tin, hoặc gain biến mất khi không teacher-force target, hoặc C2 non-ODE ngang C4 nhưng C4 chi phí cao hơn đáng kể; trong cả ba trường hợp, không tiếp tục thêm alignment/anchor modules để “cứu” formulation trước khi giải thích nguyên nhân.

---

## 15. Các câu hỏi còn mở, không tự giả định đã giải quyết

1. **User intent** nên lấy từ `h_ID` hay pooled history cả ba modality? Đề xuất `h_ID` cho MVP; đây là latent history representation, không tự động là semantic intent có thể diễn giải.
2. Score discrepancy có thể được xem là **evidence** hay chỉ là disagreement? Cần calibration, intervention, rescue/harm và source-swap controls.
3. Score-to-embedding projection `weighted sum E_receiver` có đủ để thay đổi state không, hay làm các branch homogenize? Cần theo dõi branch performance/entropy.
4. Shared vector field có làm yếu text/image? Cần separate-model baseline và modality-specific adapters.
5. **ODE đúng nghĩa** hay chỉ K-round refinement? Xác định qua conditional velocity training, state dependence, solver generalization, đối chứng non-ODE cùng budget.
6. Việc giữ full item scores mỗi step có hợp lý ở dataset lớn hơn không? Chỉ tối ưu bằng top-L sau khi có gain cơ bản và đo candidate recall.
7. Có thể giữ semantic/collaborative **complementarity** mà không yêu cầu per-modality score consistency? Đo qua rescued vs harmed items, không qua score similarity đơn lẻ.

---

## 16. One-page implementation order (khuyến nghị thực tế)

1. **Chốt ensemble baseline** của báo cáo — trước tất cả module mới.
2. **Tạo tensor item-score thống nhất** `s_m ∈ R^(B×V)` cho ID và text, với calibration / history mask đúng.
3. **Cài non-ODE iterative feedback trước** (để debug score message và có negative control mạnh).
4. **Chèn message vào per-modality velocity** của FMRec endpoint model; khởi tạo strength = 0; train rollout như inference.
5. **Không thêm KL/anchor;** chạy β/no-feedback, q/no-intent, one-way ablation.
6. **Đo mechanism**, không chỉ metric: score intervention, net rescue, state dependence.
7. **So với matched iterative non-ODE**; nếu có giá trị ODE riêng mới triển khai direct velocity/continuous-time version.
8. **Chỉ sau đó** thêm image và nhiều datasets để đánh giá novelty/publication readiness.

**Quyết định cuối:** Nên coi đây là một thử nghiệm về **cross-modal complementary evidence transfer through item scores**. ODE là một giả thuyết về *cách tổ chức quá trình refinement*, không phải tiền đề rằng K steps nhất định phải tốt hơn một step.
