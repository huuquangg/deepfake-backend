
### Feedback 1 — Abstract needs a quantitative result
**What it means:** The abstract currently ends with "results demonstrate that integrating these two types of evidence improves robustness" — a qualitative claim with no number behind it. Reviewers want at least one concrete metric so the claim is credible at a glance.

**Idea:** Rewrite the final 1–2 sentences of the Abstract to state your headline numbers, e.g. "the fused BiGRU-Attention model achieves 84.90% accuracy and 0.9369 AUC on FaceForensics++, and 83.48% accuracy and 0.9186 AUC on Celeb-DF." If you complete the ablation study (Feedback 8 below), replace this with the actual fusion gain instead — e.g. "fusion improves AUC by X points over the frequency-only baseline" — since that is the number that most directly substantiates your claim and is exactly the kind of evidence the abstract is missing.

### Feedback 2 — Introduction should state explicit, testable hypotheses
**What it means:** The Introduction currently describes motivation, related limitations, and a method overview, but never commits to a falsifiable statement like "we hypothesize that X will outperform Y." Reviewer B wants the paper framed scientifically, not just descriptively.

**Idea:** Add 2–3 explicit hypotheses as a short paragraph near the end of the Introduction (right before or after your current "We propose..." paragraph), e.g.:
- **H1:** Fusing frequency-domain forensic features with behavioral features improves detection AUC over either modality used alone.
- **H2:** Attention-based recurrent models (BiGRU/BiLSTM) generalize better than the convolutional TCN model under domain shift (FF++ → Celeb-DF).
- **H3:** HMM-based temporal smoothing improves video-level prediction consistency over raw frame-level predictions.

Then explicitly revisit each one by name in the Discussion ("H1 is supported by the ablation results in Table X..."; "H2 is supported by TCN's instability on Celeb-DF..."). This also gives the Discussion section a clearer organizing structure than it currently has.