# Slide deck brief for NotebookLM

Instructions for generating a presentation from `PAPER.docx` ("Detection of Ventricular
Tachyarrhythmias"). Target is about 15 slides. Slide 1 is the title slide, so it needs no
instruction here. Slide 15 is the conclusion. The 13 slides in between carry the story from
problem to results to what goes forward.

## Overall guidance

- Audience: a technical but general reader (an intro-to-bioinformatics course, plus engineers
  who care about real-time ECG). Assume they know basic ECG but not this specific literature.
- One idea per slide. Lead each slide with the single point it exists to make, then support it.
- Keep every number exactly as written in the paper. Do not round, invent, or soften figures
  (AUC, sensitivity, specificity, F1, cost multipliers). If a number is not in the paper, do
  not add it.
- Tone: plain, factual, confident. No hype words. This is a measurement paper, not a pitch.
- Use our own figures. They live in `paper/figures/` and each slide below names the exact
  file(s) to place on it under a `Figure:` line. Prefer these over any auto-generated visual,
  because the storyline is built around them and the numbers on them match the text. Slides
  with no `Figure:` line are text or diagram only.
- The through-line: classical deterministic features only, chosen as the feature-selection
  step toward a regulatory submission, so no machine learning is evaluated.

The ten figures:

- `fig01_nsr.png` normal sinus rhythm strip
- `fig02_vt.png` ventricular tachycardia strip
- `fig03_vfl.png` ventricular flutter strip
- `fig04_vf.png` ventricular fibrillation strip
- `fig05_screen_ranking.png` single-feature AUC and mutual-information ranking of the features
- `fig06_corr_heatmap.png` feature-feature correlation heatmap (redundancy)
- `fig07_primary_dists.png` primary-feature separation, shockable vs non-shockable
- `fig08_disc_vs_cost.png` discrimination against compute cost, all candidates
- `fig09_roc.png` ROC curves of the tuned detectors
- `fig10_vfl_vs_vf.png` flutter vs fibrillation on the best split feature

## Per-slide focus

**Slide 2, the problem.** What ventricular tachyarrhythmias are and why they kill. VT, VFL,
and VF form a continuum of increasingly disorganised ventricular activity. VF is involved in
about 75% of sudden cardiac arrest cases at the onset. Show the four example strips side by
side so the audience sees the morphology break down from a normal beat to chaos. This visual
comparison is the single most important image in the deck.
Figure: `fig01_nsr.png`, `fig02_vt.png`, `fig03_vfl.png`, `fig04_vf.png` (all four, in this
order, on one slide).

**Slide 3, the decision that matters.** Devices do not need a diagnosis, they need one binary
call: shockable versus non-shockable. Explain that VT, VFL, and VF are the shockable side and
everything else is non-shockable, and that an AED or implantable device has to make this call
in a few seconds on a short segment. This frames every result that follows.

**Slide 4, why it is hard.** During VF and VFL there is no clean P-QRS-T to lock onto, so beat
detection fails exactly when it is needed. Morphologies overlap, artifacts imitate VF, and
flutter sits on the border between organised and disorganised. Detection has to work on the
raw signal shape, not on beats.

**Slide 5, aim and scope.** State the exact question: which classical, deterministic
signal-processing features best separate shockable from non-shockable rhythms, and can the
strongest be tuned into a usable detector. Stress the scope decision: deterministic by design,
as the feature-selection step toward a regulatory submission, so learned classifiers are
reviewed as future work, not evaluated.

**Slide 6, the candidates and feature families.** Introduce the five candidate detectors:
TCSC, VFLEAK, SPEC, HILB, and JEKOVA. Group them by the signal property they exploit
(threshold crossing, spectral concentration, complexity and phase space, band-pass counting).
This is the menu that the rest of the paper narrows down.

**Slide 7, data and labelling.** Four PhysioNet databases (VFDB, CUDB, AHADB, MITDB). Each
record is filtered once, then sliced into overlapping windows at 8 and 4 seconds. Windows are
labelled shockable or non-shockable by the benchmark convention. Sixteen signal-only features
come from vftx, a validated reimplementation of the 27 features in Hong's thesis. Note that
QRS-based and slow EMD features are excluded by design.

**Slide 8, how candidates are judged.** Two axes, not one. Discrimination (how well a feature
separates the two classes) and compute cost (how expensive it is to run), because the target
is a real-time embedded detector. Name the discrimination metrics briefly: single-feature AUC,
mutual information, and best F1 at a swept threshold. The whole method is screen, then shootout,
then tune the leaders, then a flutter-versus-fibrillation check.

**Slide 9, feature screen.** Result one: the JEKOVA 14.6 Hz band-pass counts lead the screen,
with single-feature AUC 0.986. Show the feature ranking, then the separation the top feature
gives between the two classes. The point: one cheap band-pass family carries most of the
separation. If the redundancy heatmap helps, add it, but the ranking and the separation plot
are the priority.
Figure: `fig05_screen_ranking.png` and `fig07_primary_dists.png` (ranking plus separation);
`fig06_corr_heatmap.png` optional if there is room.

**Slide 10, the shootout, discrimination against cost.** The central trade-off slide. Use the
discrimination-versus-cost figure. JEKOVA wins on discrimination; TCSC is about fourteen times
cheaper and comes second on discrimination (F1 0.706). Conclusion: there is no single winner,
so both go forward, one accuracy-first (JEKOVA) and one cost-first (TCSC).
Figure: `fig08_disc_vs_cost.png`.

**Slide 11, tuned detectors.** Turning features into detectors. Tuning JEKOVA's published
decision cascade reaches F1 0.847 (sensitivity 0.898, specificity 0.976). The reproduced
published cascade gives sensitivity 0.973 and specificity 0.900, close to the original paper,
which is an independent check that the reimplementation is faithful. Present the two operating
points (accuracy-first and cost-first) as the deliverable.
Figure: `fig09_roc.png`.

**Slide 12, flutter versus fibrillation.** An honest negative result. The winning feature does
not separate flutter from fibrillation (AUC 0.603), because both share the same band-pass
signature. Spectral and regularity features do a little better. Flutter is rare in the data, so
this is indicative, not settled.
Figure: `fig10_vfl_vs_vf.png`.

**Slide 13, discussion and limits.** Tie it together: the accuracy-first versus cost-first
choice is a real engineering decision, not a tie to break. Note that the results line up with
the published literature, and state the main threats to validity (database imbalance, window
labelling, flutter rarity) plainly.

**Slide 14, AI as a research collaborator.** The paper's methodological chapter. A grounded,
closed-world setup with a phased, reviewed process, explicit safeguards against fabricated
citations and numbers, and human authorship and responsibility retained. Keep this to the
process and the safeguards, not a testimonial.

**Slide 15, conclusion and future work.** Close the loop. Two deterministic detectors go
forward, JEKOVA for accuracy and TCSC for cost, as the feature-selection step toward a
regulatory submission. Future work: a majority-vote episode layer to score by duration, and
learned classifiers evaluated against these deterministic baselines. End on the one sentence a
listener should remember: cheap classical features already separate shockable from
non-shockable rhythms well enough to build on.
