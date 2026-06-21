## Reviewer B

### Feedback 1 — Abstract needs a quantitative result (Quang will update this)
**What it means:** The abstract currently ends with "results demonstrate that integrating these two types of evidence improves robustness" — a qualitative claim with no number behind it. Reviewers want at least one concrete metric so the claim is credible at a glance.

**Plan to update:** Rewrite the final 1–2 sentences of the Abstract to state your headline numbers, e.g. "the fused BiGRU-Attention model achieves 84.90% accuracy and 0.9369 AUC on FaceForensics++, and 83.48% accuracy and 0.9186 AUC on Celeb-DF." If you complete the ablation study (Feedback 8 below), replace this with the actual fusion gain instead — e.g. "fusion improves AUC by X points over the frequency-only baseline" — since that is the number that most directly substantiates your claim and is exactly the kind of evidence the abstract is missing.

---

### Feedback 2 — Introduction should state explicit, testable hypotheses (AI will review and update this)
**What it means:** The Introduction currently describes motivation, related limitations, and a method overview, but never commits to a falsifiable statement like "we hypothesize that X will outperform Y." Reviewer B wants the paper framed scientifically, not just descriptively.

**Plan to update:** Add 2–3 explicit hypotheses as a short paragraph near the end of the Introduction (right before or after your current "We propose..." paragraph), e.g.:
- **H1:** Fusing frequency-domain forensic features with behavioral features improves detection AUC over either modality used alone.
- **H2:** Attention-based recurrent models (BiGRU/BiLSTM) generalize better than the convolutional TCN model under domain shift (FF++ → Celeb-DF).
- **H3:** HMM-based temporal smoothing improves video-level prediction consistency over raw frame-level predictions.

Then explicitly revisit each one by name in the Discussion ("H1 is supported by the ablation results in Table X..."; "H2 is supported by TCN's instability on Celeb-DF..."). This also gives the Discussion section a clearer organizing structure than it currently has.

---

### Feedback 3 — HMM state-to-label mapping is a heuristic, not a learned/inference-ready component
**What it means:** Your own Implementation Details section already admits the HMM hidden-state-to-label mapping is done via majority voting using ground-truth labels per fold. The reviewer is flagging that this isn't something you could actually do at real deployment time (you don't have ground truth at inference), so the reported HMM-smoothed numbers may be somewhat optimistic versus a true end-to-end system.

**Plan to update:** (Q&A: Do we need to retrain again with this? If yes)
1. Report results **with** and **without** HMM smoothing side by side (a small additional table, or extra columns in Table `tab:main_results`), so readers can see exactly how much the heuristic step contributes.
2. In the Discussion, beyond just acknowledging the limitation (which you already do), propose a concrete deployment-ready fix — e.g., "fix the state-to-label mapping using the global mean predicted probability per state on the training set only, then freeze it for inference," rather than re-deriving the mapping per fold from ground truth. Stating the fix, even if not fully re-run, shows the limitation is understood at a solvable level rather than just flagged.

---

### Feedback 4 — Figures and tables need higher resolution and self-contained captions
**What it means:** This is a production-quality note: the architecture figures (`BIGRU_FINAL.jpg`, `bilstm_kan_architecture.png`, `tcn_architecture.png`) need to be legible at print resolution, and every caption should let a reader understand the figure without referring back to the body text.

**Plan:**
1. Re-export all three architecture images at ≥300 dpi or as vector graphics (PDF/SVG) before camera-ready submission — JPEG at low resolution is the most likely culprit here.
2. Expand the captions for Fig. 2 (BiGRU), Fig. 3 (BiLSTM-KAN), and Fig. 4 (TCN) to match the level of detail you already gave Fig. 1 (the pipeline diagram) — i.e., briefly state what the diagram shows and what it's meant to demonstrate, not just a one-line title.
3. Double-check the TikZ pipeline diagram (Fig. 1) renders cleanly at single-column width in the final compiled PDF, since TikZ figures can shift or clip differently across LaTeX engines.

---

## Reviewer C

### Feedback 5 — "The primary concern is the limited demonstration of novelty. Each of these components has been extensively explored in prior deepfake detection literature."
**What it means:** FFT/DCT/SRM forensic features, OpenFace behavioral cues, MobileNetV2 backbones, BiGRU/BiLSTM/TCN, and HMM smoothing are all individually well-established techniques the reviewer has seen before elsewhere. As written, the paper reads as "we combined known components" without making clear what new insight or capability this combination produces.

**Plan:** Add a short, numbered "Contributions" paragraph at the end of the Introduction that states precisely what is new, separate from what is borrowed. Don't just describe the pipeline again — frame each item as a claim, e.g.: "Unlike prior work that uses frequency-forensic or behavioral signals in isolation, we are the first to combine FFT+DCT+SRM with OpenFace behavioral cues in a single temporal fusion framework targeted at mobile face-authentication scenarios." This paragraph is also what Feedbacks 6 and 7 below will feed into directly, so it's worth writing after you've finalized your ablation and baseline results.

---

### Feedback 6 — "What is the contribution of the authors?"
**What it means:** A direct, almost rhetorical question — the reviewer cannot identify, in plain language, what part of the paper is "yours" versus assembled from existing tools (MobileNetV2, OpenFace, standard FFT/DCT/SRM extraction, standard RNN/TCN architectures, standard HMM).

**Plan:** Answer this question explicitly and honestly in the same Contributions paragraph from Feedback 5. Be specific about the nature of the contribution — if it is primarily an empirical/systems contribution (a controlled comparative study under a novel feature combination) rather than a new algorithmic component, say so directly rather than implying architectural novelty you can't back up. Reviewers respond far better to an honestly scoped claim than to one that overstates novelty and gets caught.

---

### Feedback 7 — "The manuscript does not clearly explain what is fundamentally novel beyond combining existing modules, and whether the contribution is architectural or application-oriented."
**What it means:** This is asking you to classify your own contribution. Is the paper proposing a new architecture (a new fusion mechanism, a new layer type), or is it an applied/benchmarking contribution (showing that combining known modules works well for a specific use case — mobile banking face authentication)? Right now the paper doesn't say.

**Plan:** State this classification explicitly in the Contributions paragraph: e.g., "Our contribution is primarily application-oriented and empirical: we are the first to benchmark these three temporal fusion backbones under an identical dual-stream (forensic + behavioral) feature set, and to quantify the individual contribution of each modality (see Section IV-C) in a mobile-authentication-relevant evaluation setting." This directly resolves the ambiguity the reviewer is pointing at — you're not claiming a new architecture, you're claiming a new, validated combination with measured contribution.

---

### Feedback 8 — "The paper lacks sufficient architectural details: complete architecture diagram, layer-wise specification table, mathematical formulation of feature fusion."
**What it means:** Three distinct asks bundled in one sentence, all about reproducibility-level detail:
1. A full architecture diagram (not just the current block-level pipeline figure).
2. A layer-by-layer specification table (layer types, output shapes, units, activations).
3. An explicit equation for how the spatial and forensic/behavioral streams are fused.

**Plan:**
1. **Layer-wise tables:** Add one specification table per temporal architecture (3 tables total: BiGRU+Attention, BiLSTM+KAN, TCN) in §III-B, with columns Layer / Output Shape / Units-Filters / Activation / Parameter Count.
2. **Fusion equation:** Add an explicit formula where the "Late fusion" box currently sits in Fig. 1, e.g.:
   $$x_t = \big[\,\text{MobileNetV2}(I_t);\ \text{Norm}(\text{FFT}_t,\text{DCT}_t,\text{SRM}_t,\text{OpenFace}_t)\,\big] \in \mathbb{R}^{D_s+D_f}$$
   with $D_s$ and $D_f$ replaced by your actual implementation's dimensions (e.g., $D_s=1280$ for MobileNetV2's GAP output).
3. **Architecture diagram:** Expand Fig. 1 (or add a new figure) to show actual tensor shapes flowing between blocks, not just named stages — this turns it from a conceptual pipeline diagram into a true architecture diagram the reviewer can use to reproduce your system.

---

### Feedback 9 — "Why were FFT, DCT, and SRM selected instead of Wavelet Transform or Discrete Wavelet Packet Transform? Please provide justification supported by literature or experimental support."
**What it means:** This is a direct challenge to a specific design decision. Wavelet-based methods (DWT, DWPT) are a recognized competing family for multi-resolution frequency-artifact detection, and the reviewer wants to know why you didn't use them instead — backed by either citation or your own experiment.

**Plan:** Two options, depending on time:
- **Literature-only justification (minimum):** Add a short paragraph (Related Work §II-C or Methodologies) explaining that FFT/DCT were chosen because they align directly with the actual JPEG/H.264 compression pipeline used by both datasets — DCT's 8×8 blocks literally match the encoder's transform, so DCT-domain anomalies correspond directly to compression-stage artifacts, whereas wavelet transforms have no such direct correspondence to the codecs in use. Pair this with the established precedent of SRM specifically in steganalysis/forensics (Fridrich & Kodovský) as the standard residual-domain approach used in deepfake detection literature.
- **Experimental justification (stronger):** If time allows, run one additional ablation fold replacing FFT+DCT with a DWT-based feature set (e.g., 2–3 level Daubechies decomposition via PyWavelets) under the same BiGRU pipeline on FF++, and report the AUC/accuracy difference. Even a single-fold result lets you say "we additionally tested DWT and observed X," which is far more persuasive to this reviewer than citation alone.

---

### Feedback 10 — "The paper proposes a dual-stream framework but does not quantify the contribution of individual components. How much performance improvement is obtained by adding behavioral features compared to using only frequency-domain features?"
**What it means:** This is the most concrete and important gap in the paper. You claim fusion helps, but there is currently no experiment that isolates each modality's contribution — no frequency-only run, no behavioral-only run, nothing to compare the full fusion model against.

**Plan:** Add a new **Ablation Study** subsection in Experiments, immediately after Main Results. Using your best-performing backbone (BiGRU+Attention), train and evaluate three variants:
1. Spatial + Frequency-only (drop OpenFace features from the fused vector).
2. Spatial + Behavioral-only (drop FFT/DCT/SRM features).
3. Full fusion (current model, for comparison).

Report the same five metrics (Accuracy, Precision, Recall, F1, AUC) for all three variants on both FF++ and Celeb-DF — 3-fold cross-validation is acceptable here to limit compute cost. This single table directly answers this comment, supplies the fusion-gain number Reviewer B wants in the abstract (Feedback 1), and is the strongest evidence you can offer for the novelty/contribution questions (Feedbacks 5–7).

---

### Feedback 11 — "Limited Experimental analysis on benchmark models."
**What it means:** You currently only compare your three proposed temporal backbones against each other — there is no comparison against any existing published deepfake detector (e.g., MesoNet, a plain Xception/CNN baseline, FreqNet). Without this, a reader can't tell whether your numbers are even competitive with the field.

**Plan:** Add a baseline comparison table in Experiments, including at minimum:
1. A simple spatial-only baseline (MobileNetV2 alone, no frequency/behavioral fusion, no temporal model) trained on your own data splits — cheap to produce since this branch already exists in your pipeline.
2. Published numbers from MesoNet~\cite{b1} and/or FreqNet~\cite{b5} on FF++/Celeb-DF, clearly cited and explicitly labeled "as reported in the original paper" with a caveat that protocols/splits differ, so you're not implying a strictly fair apples-to-apples comparison where one doesn't exist.

---

### Feedback 12 — "Analyze the computational complexity of proposed model."
**What it means:** Since your stated motivation is mobile banking deployment, the reviewer wants evidence the model is actually feasible on that target — parameter count, FLOPs, and inference latency, not just accuracy metrics.

**Plan:** Add a short **Computational Complexity** subsection (Experiments or Discussion) reporting, for the MobileNetV2 backbone and each of the three temporal models separately:
- Total trainable parameters (directly available via `model.summary()`/`count_params()`).
- Approximate FLOPs per forward pass.
- Measured inference latency per video sequence (ms), ideally both GPU and CPU-only, since mobile devices often lack GPU acceleration — this ties directly back to your own stated mobile-deployment motivation.
- A brief sentence contrasting TCN's parallelizable inference against BiGRU/BiLSTM's sequential inference — a real practical tradeoff your Methodologies section already gestures at qualitatively (§III-B.3) but never quantifies.

---

### Feedback 13 — "Show some test results."
**What it means:** Ambiguous as phrased, but most plausibly asking for qualitative/visual evidence beyond aggregate metrics — actual example predictions, confusion matrices, or visualizations that make the results tangible rather than purely numeric.

**Plan:**
1. Fill in a **pooled confusion matrix** per dataset for your best model — you already have placeholder LaTeX tables for this commented out in the source; populate them with real counts from your evaluation runs.
2. Add 2–4 qualitative example cases: a correctly flagged fake video frame with its predicted probability, and where feasible, a visualization of the attention weights across the 10/15-frame sequence showing which frames triggered detection. This ties back nicely to your existing claim in §III-B.1 that multi-head attention identifies "betraying" frames — showing it in practice is far more convincing than asserting it.