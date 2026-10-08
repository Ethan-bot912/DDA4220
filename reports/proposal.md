# Deep Learning for Automated Building Damage Assessment from Paired Satellite Imagery

**DDA4220 Project Proposal**

**Project type:** Application-oriented project with a controlled empirical study

**Team member:** BIAN Yining 124090006
                 LIU Zhaocheng 124090403
                 WANG Kexin 124090614
                 WANG Mian 124090618

**Repository:** [DDA4220](https://github.com/Ethan-bot912/DDA4220)

## 1. Problem and objective

Natural disasters can damage many buildings across a large area. Emergency-response teams need timely estimates of where damage has occurred and how severe it is, while conventional response can depend on in-person assessment [1]. This project will study whether paired pre-disaster and post-disaster images can support faster, wide-area building-level damage assessment.

The central research question is:

> Does pre-disaster context improve building-level damage classification over a credible post-disaster-only baseline, and is any improvement worth the additional computational cost?

We will use the four xBD damage categories: `no damage`, `minor damage`, `major damage`, and `destroyed` [1]. The core project will compare two Stage B settings on identical building crops derived from ground-truth polygons: a post-disaster-only baseline and a paired pre- and post-disaster classifier. This isolates the practical value of adding pre-disaster context to the damage classifier from errors in building localization. If time and computing resources permit, we will add a localization stage and measure how predicted building footprints affect the complete pipeline.

The project is an application and empirical study, not a claim of a new architecture. Its value will come from a controlled comparison, a reproducible data protocol, and an analysis of when paired imagery helps or fails.

## 2. Planned readings and starting point

We will examine the following resources before fixing the final experimental design.

**Table 1. Planned readings and their roles in the project.**

| Reading or resource | Why it is relevant |
|---|---|
| Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery* [1] | To understand the paired-image structure, polygon and damage labels, disaster metadata, evaluation setting, and known dataset limitations. |
| Weber and Kané, *Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion* [2] | To examine the closest prior work on mono-temporal versus multi-temporal xBD models, data processing, and practical ways to incorporate pre-disaster context. |
| Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* [3] | To provide a practical starting method for the optional building-localization stage. |
| He et al., *Deep Residual Learning for Image Recognition* [4] | To motivate a standard CNN encoder for the damage classifiers and a comparable backbone across the two input settings. |
| Official xView2 data and baseline resources [5, 6] | To verify data access, file formats, preprocessing conventions, and the reference localization-classification pipeline. |

Our first implementation will be a small, complete post-disaster-only classifier using ground-truth building crops and a ResNet-18 encoder. This is both a credible baseline and a feasibility check: it will expose data-loading problems and provide initial measurements of training time and memory before we implement the paired-image classifier.

## 3. Planned data and implementation

### 3.1 Data protocol

We plan to use the xBD dataset released for the xView2 challenge, which contains paired pre- and post-disaster satellite images, building polygons, four-level damage labels, and disaster-event metadata [1, 5]. We will begin with a documented subset of approximately three to five disaster events and expand it only after a complete pilot run is stable.

Before training, we will verify access to the images, labels, and relevant baseline code; inspect real image pairs and polygon overlays; record label frequencies and alignment issues; and save a manifest of the selected samples. To reduce geographic and event leakage, train, validation, and test partitions will be grouped by disaster event rather than created by randomly splitting individual buildings. The held-out test partition will not be used for model selection.

Ground-truth polygons will define building crops with a fixed context margin. Every classifier will use the same crop coordinates, resolution, split, label mapping, and paired spatial augmentations. Class weights will be computed from the training partition to address damage-class imbalance.

### 3.2 Controlled Stage B comparison

The controlled classification study will include two primary settings:

- **B1 - post-only baseline:** a CNN predicts damage from the post-disaster crop.
- **B2 - paired pre/post classifier:** a CNN predicts damage using the aligned pre- and post-disaster crops of the same building.

B1 is the main credible baseline, while B2 tests whether adding pre-disaster context improves damage classification. The two settings will use the same data split, building crops, preprocessing, backbone family, training budget, and checkpoint-selection protocol. Their primary intended difference is whether Stage B receives the pre-disaster building crop. B2 will use a straightforward paired-input fusion method that is fixed before final test evaluation. Comparing multiple input-level, feature-level, or Siamese fusion architectures is outside the primary scope of this application-oriented project; if time and computing resources permit, alternative fusion methods may be studied only as supplementary ablations using the training and validation data.

As a stretch objective, a U-Net-style model will segment building footprints from pre-disaster imagery [3]. The selected damage classifier will then be evaluated using crops derived from predicted footprints. Results from ground-truth and predicted footprints will be reported separately so that localization error is not confused with classification error.

## 4. Evaluation and expected contribution

The primary classification metric will be **Macro F1**, because overall accuracy can hide poor performance on less frequent severe-damage classes. We will also report balanced accuracy, per-class precision, recall and F1, and a confusion matrix. The primary experiment will compare B2 with B1 on the same held-out buildings, directly testing whether adding the pre-disaster crop to Stage B improves damage classification. If the optional localization stage is completed, it will be evaluated with intersection over union and Dice/F1.

Both primary settings will share the same data split, preprocessing, training budget, and validation-based checkpoint rule. Where resources allow, the main comparison will use at least three fixed random seeds and report mean and standard deviation; otherwise, the single-seed limitation will be stated. We will record parameter count, peak memory, and inference latency on the same hardware, alongside prediction quality, so that any benefit from pre-disaster context can be weighed against its additional computational cost. Error analysis will examine damage class, disaster event, image alignment, shadows, small buildings, and ambiguous labels.

The expected core contribution is a reproducible satellite-image application for building-level damage classification, supported by a controlled comparison of post-disaster-only and paired pre/post settings on an event-aware xBD subset. The study will show whether adding the pre-disaster crop improves prediction, under which conditions it helps, and whether the gain justifies its computational overhead, rather than claiming a new fusion architecture. If the optional localization stage is completed, we will also estimate the performance loss caused by predicted rather than ground-truth building footprints.

## 5. Feasibility, risks, and next steps

The core scope is intentionally limited to the controlled building-crop comparison. Localization and the end-to-end damage map are extensions rather than requirements for a successful project.

**Table 2. Main feasibility risks and mitigation measures.**

| Risk | Mitigation |
|---|---|
| Data access or scale prevents a full experiment | Verify access first, inspect actual examples, begin with three to five events, and preserve a reproducible manifest. If full access fails, use a verified official subset or a documented reduced experiment rather than silently substituting another dataset. |
| Pre/post misalignment weakens the temporal comparison | Inspect overlays, apply spatial transforms jointly, record invalid pairs, and include alignment-related error analysis. |
| Class imbalance hides failure on severe damage | Use training-only class weights and report Macro F1 and per-class results rather than accuracy alone. |
| The paired classifier exceeds available GPU time, memory, or queue capacity | Estimate cost with a small complete run, use a compact backbone, reserve scheduling buffer, and prioritize the post-only versus paired comparison over optional fusion ablations and the stretch pipeline. |

The team's immediate next steps are:

1. Read the selected papers and verify xBD data and baseline-code access.
2. Inspect paired samples, create the event-aware split, and run one complete B1 pilot.
3. Fix the preprocessing and evaluation protocol, then implement the paired pre/post classifier with a straightforward fusion method.
4. Complete the primary post-only versus paired comparison and error analysis; consider fusion ablations only if resources remain, before deciding whether to add the optional localization stage.


## References

[1] R. Gupta et al., "xBD: A Dataset for Assessing Building Damage from Satellite Imagery," *arXiv preprint arXiv:1911.09296*, 2019. <https://arxiv.org/abs/1911.09296>

[2] E. Weber and H. Kané, "Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion," in *ICLR 2020 Workshop on AI for Earth Sciences*, 2020. <https://arxiv.org/abs/2004.05525>

[3] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *MICCAI*, 2015. <https://arxiv.org/abs/1505.04597>

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *CVPR*, 2016. <https://arxiv.org/abs/1512.03385>

[5] xView2, "xView2: Assess Building Damage," data and project resources. <https://xview2.org/>

[6] DIUx-xView, "xView2 Baseline: Baseline Localization and Classification Models for the xView2 Challenge." <https://github.com/DIUx-xView/xView2_baseline>
