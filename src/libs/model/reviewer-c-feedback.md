### Feedback 5 — "The primary concern is the limited demonstration of novelty. Each of these components has been extensively explored in prior deepfake detection literature."

**What it means:** FFT/DCT/SRM forensic features, OpenFace behavioral cues, MobileNetV2 backbones, BiGRU/BiLSTM/TCN, and HMM smoothing are all individually well-established techniques the reviewer has seen before elsewhere. As written, the paper reads as "we combined known components" without making clear what new insight or capability this combination produces.

**Idea:** Add a short, numbered "Contributions" paragraph at the end of the Introduction that states precisely what is new, separate from what is borrowed. Don't just describe the pipeline again — frame each item as a claim, e.g.: "Unlike prior work that uses frequency-forensic or behavioral signals in isolation, we are the first to combine FFT+DCT+SRM with OpenFace behavioral cues in a single temporal fusion framework targeted at mobile face-authentication scenarios." This paragraph is also what Feedbacks 6 and 7 below will feed into directly, so it's worth writing after you've finalized your ablation and baseline results.

---

### Feedback 6 — "What is the contribution of the authors?"

**What it means:** A direct, almost rhetorical question — the reviewer cannot identify, in plain language, what part of the paper is "yours" versus assembled from existing tools (MobileNetV2, OpenFace, standard FFT/DCT/SRM extraction, standard RNN/TCN architectures, standard HMM).

**Idea:** Answer this question explicitly and honestly in the same Contributions paragraph from Feedback 5. Be specific about the nature of the contribution — if it is primarily an empirical/systems contribution (a controlled comparative study under a novel feature combination) rather than a new algorithmic component, say so directly rather than implying architectural novelty you can't back up. Reviewers respond far better to an honestly scoped claim than to one that overstates novelty and gets caught.

---

### Feedback 7 — "The manuscript does not clearly explain what is fundamentally novel beyond combining existing modules, and whether the contribution is architectural or application-oriented."

**What it means:** This is asking you to classify your own contribution. Is the paper proposing a new architecture (a new fusion mechanism, a new layer type), or is it an applied/benchmarking contribution (showing that combining known modules works well for a specific use case — mobile banking face authentication)? Right now the paper doesn't say.

**Idea:** State this classification explicitly in the Contributions paragraph: e.g., "Our contribution is primarily application-oriented and empirical: we are the first to benchmark these three temporal fusion backbones under an identical dual-stream (forensic + behavioral) feature set, and to quantify the individual contribution of each modality (see Section IV-C) in a mobile-authentication-relevant evaluation setting." This directly resolves the ambiguity the reviewer is pointing at — you're not claiming a new architecture, you're claiming a new, validated combination with measured contribution.

### Feedback 9 — "Why were FFT, DCT, and SRM selected instead of Wavelet Transform or Discrete Wavelet Packet Transform? Please provide justification supported by literature or experimental support." @Quang D Huu

**What it means:** This is a direct challenge to a specific design decision. Wavelet-based methods (DWT, DWPT) are a recognized competing family for multi-resolution frequency-artifact detection, and the reviewer wants to know why you didn't use them instead — backed by either citation or your own experiment.

**Idea:**
- **Literature-only justification (minimum):** Add a short paragraph (Related Work §II-C or Methodologies) explaining that FFT/DCT were chosen because they align directly with the actual JPEG/H.264 compression pipeline used by both datasets — DCT's 8×8 blocks literally match the encoder's transform, so DCT-domain anomalies correspond directly to compression-stage artifacts, whereas wavelet transforms have no such direct correspondence to the codecs in use. Pair this with the established precedent of SRM specifically in steganalysis/forensics (Fridrich & Kodovský) as the standard residual-domain approach used in deepfake detection literature. I think citation-based justification should be sufficient. We do not need to rerun the experiments with Wavelet/DWPT. Instead, we should make the literature-supported rationale for selecting FFT, DCT, and SRM more convincing, especially by explaining why these features are more aligned with compression and forensic artifacts in our datasets.