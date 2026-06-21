# Research Skill: Finding Precise, High-Quality Sources for Deepfake Detection Papers

A repeatable workflow for turning reviewer feedback into targeted literature additions — web, journals, conferences, and books — for deepfake/synthetic-media-detection research.

---

## 0. Start from the feedback, not from a blank search bar

Before searching anything, convert each piece of reviewer feedback into a **specific, answerable question**. Vague searching wastes time; specific gaps find exact papers.

| Reviewer comment (example) | Turn it into a search question |
|---|---|
| "Related work is thin" | "What are the SOTA deepfake detectors 2024–2026 on Celeb-DF v2 / DFDC?" |
| "Doesn't address generalization" | "Cross-dataset generalization failure modes in CNN-based deepfake detectors" |
| "Missing diffusion-based fakes" | "Detection methods for diffusion-generated/diffusion-edited faces" |
| "No discussion of audio deepfakes" | "ASVspoof / audio deepfake detection benchmarks and methods" |
| "Needs stronger baseline comparison" | "Benchmark comparison tables: Xception, EfficientNet, ViT detectors on FF++" |

Keep a running table like this — it becomes your literature-review checklist later.

---

## 1. Where to search (in priority order for this field)

1. **Google Scholar** — broadest coverage, citation counts, "Cited by" and "Related articles" links.
2. **Semantic Scholar** (semanticscholar.org) — AI-generated TL;DRs, citation graphs, influential-citation filter (very useful for separating impactful work from noise).
3. **arXiv** (arxiv.org, categories `cs.CV`, `cs.MM`, `cs.CR`, `eess.AS` for audio) — fastest-moving subfield; most deepfake work appears here before/alongside formal publication.
4. **Papers With Code** (paperswithcode.com) — leaderboard view by dataset (e.g., search "DFDC", "FaceForensics++", "Celeb-DF") — instantly shows current SOTA and links to code + paper.
5. **DBLP** (dblp.org) — clean, deduplicated bibliography of CS venues; good for checking a specific author's or venue's output.
6. **Connected Papers / Litmaps** — visual citation-graph tools; paste one strong seed paper in and it shows the surrounding research cluster (great for "snowballing," see §3).
7. **IEEE Xplore** and **ACM Digital Library** — for TIFS, TPAMI, ACM MM papers not always mirrored on arXiv.
8. Avoid relying on general web search or blog posts for technical claims — use them only to find leads, then verify against the actual paper.

---

## 2. Key venues to know for this subfield

Treat these as your "trusted venue" list when judging whether a source is rigorous:

- **Top-tier vision/ML conferences:** CVPR, ICCV, ECCV, NeurIPS, ICML, AAAI, WACV, BMVC
- **Multimedia/forensics-specific:** ACM Multimedia (ACM MM), IEEE WIFS (Workshop on Information Forensics and Security), IJCB
- **Audio deepfakes:** Interspeech, ICASSP, ASVspoof Challenge proceedings
- **Journals:** IEEE Transactions on Information Forensics and Security (TIFS), IEEE TPAMI, Pattern Recognition, ACM Computing Surveys (for review papers)
- **Surveys to anchor your related-work section:** search Google Scholar for `"deepfake detection" survey OR review 2024..2026` — a recent survey paper's reference list is one of the fastest ways to find 30–50 relevant citations at once.

---

## 3. Citation snowballing (the single most efficient technique)

1. Pick 1–3 strong seed papers (recent, highly cited, closely related to your method).
2. **Backward snowball:** read their reference lists for foundational work you're missing.
3. **Forward snowball:** use Google Scholar's "Cited by" or Semantic Scholar's citation graph to find newer papers that cite your seed — this surfaces work published *after* your seed, which is exactly what's likely missing if your draft cites older baselines.
4. Repeat 1–2 rounds. Stop when new searches mostly return papers you've already found (saturation).

---

## 4. Datasets and benchmarks worth knowing by name

Knowing these lets you write more precise queries and evaluate whether a paper's claims are tested on a credible benchmark:

- **Face forgery/video:** FaceForensics++ (FF++), DFDC (Deepfake Detection Challenge), Celeb-DF (v1/v2), DeeperForensics-1.0, WildDeepfake, KoDF, FaceShifter
- **Diffusion/GAN-generated images:** DiffusionDB, GenImage, synthetic-face-detection benchmarks built on Stable Diffusion / StyleGAN outputs
- **Audio:** ASVspoof (2019/2021/2024), FakeAVCeleb (audio-visual)
- When you find a new paper, check **which dataset(s)** it reports on — this tells you immediately whether it's directly comparable to your own evaluation.

---

## 5. Query construction tips

- Use field-specific synonyms together: `deepfake OR "face forgery" OR "face manipulation" OR "synthetic media" OR "GAN-generated" OR "diffusion-generated"`
- Combine with your specific angle: `+ "generalization" `, `+ "frequency domain"`, `+ "temporal artifacts"`, `+ "explainability"`, `+ "compression robustness"`
- Add a date filter (Scholar/arXiv allow this) to isolate **2024–2026** work specifically if a reviewer flagged outdated references.
- For arXiv, you can search by category + keyword directly: `https://arxiv.org/list/cs.CV/recent` or use arXiv's own search with `abs:"deepfake detection"`.

---

## 6. Verifying quality and avoiding bad citations

- Prefer peer-reviewed venue versions over arXiv preprints when both exist (cite the published version).
- Check **venue tier** against the list in §2; an unfamiliar venue isn't automatically weak, but verify it (check CORE conference ranking or journal impact factor if unsure).
- Be cautious with preprints that have **zero citations and no code release** in a fast-moving field — they may be unverified or superseded.
- Cross-check any quantitative claim (e.g., "achieves 99% accuracy on DFDC") against at least one other source (a survey, a leaderboard, or the paper's own GitHub) before repeating it in your revision.
- Note publication date next to every source you log — this field moves fast, and reviewers often specifically flag stale citations.

---

## 7. Organize as you go

Keep a simple tracking table (spreadsheet or Zotero with tags) with columns:

`Reviewer point addressed | Citation | Venue/Year | Dataset(s) used | One-line relevance note | Where it goes in the paper`

This does two things: prevents duplicate searching, and gives you a ready-made map for inserting citations into the exact sections the reviewers flagged.

---

## 8. Step-by-step workflow checklist

- [ ] Convert every reviewer comment into a specific search question (§0)
- [ ] Find 1–3 seed papers per question via Google Scholar / Semantic Scholar / Papers With Code
- [ ] Snowball forward and backward from each seed (§3)
- [ ] Check Papers With Code leaderboards for the relevant dataset to confirm current SOTA
- [ ] Pull a recent survey paper for broad coverage and additional reference leads
- [ ] Verify venue/quality and publication date for each candidate citation (§6)
- [ ] Log each accepted source in your tracking table (§7) with the reviewer point it resolves
- [ ] Paraphrase all findings in your own words when writing — never copy text directly from sources

---

### Note
If you'd like, I can run a live search right now on one of your specific gaps (e.g., "diffusion-based deepfake detection 2025–2026") and bring back actual candidate papers instead of just the method.