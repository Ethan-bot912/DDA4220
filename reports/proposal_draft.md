# Deep Learning for Automated Building Damage Assessment from Paired Satellite Imagery

**Project proposal draft**  
**Project type:** Application-oriented project with a controlled empirical study  
**Team size:** Four members  
**Working repository:** DDA4220 project

## 1. Project summary

Natural disasters can damage a large number of buildings over a wide area. Rapid damage assessment is useful for emergency response, but manual interpretation of high-resolution satellite images is slow, expensive, and difficult to scale. This project will develop a reproducible deep-learning pipeline that uses paired pre-disaster and post-disaster satellite images to locate buildings and classify the damage level of each building.

The central empirical question is:

> Does pre-disaster context improve building-level damage classification compared with using the post-disaster image alone, and what is the cost of using the additional image?

The project is deliberately scoped as an application and empirical study rather than a claim of a new neural architecture. We will adapt established segmentation and image-classification methods to the xBD/xView2 task, establish a well-trained post-only baseline, and compare it with progressively stronger ways of using the pre-disaster image. The main contribution will be a controlled comparison, a reproducible data and evaluation protocol, and an analysis of when paired imagery helps or fails.

The planned system has two stages:

1. **Stage A - building localization:** segment building regions from the post-disaster image.
2. **Stage B - damage classification:** classify each building as `no damage`, `minor`, `major`, or `destroyed` using either the post-disaster crop or the paired pre/post crops.

To separate the scientific question from segmentation errors, the project will first evaluate Stage B using ground-truth building polygons. The full pipeline using predicted building masks will then be added if the data and compute budget allow. This creates a useful minimum project even if the end-to-end extension has to be reduced.

## 2. Problem and motivation

### 2.1 Problem definition

For a disaster event, let `I_pre` and `I_post` denote aligned pre-disaster and post-disaster satellite images. Each building has a polygon `P_i` and a damage label `y_i` from four ordered categories:

`no damage`, `minor`, `major`, and `destroyed`.

The system should produce either:

- a building mask `M` for the localization task; or
- a set of building-level predictions `(P_i, y_hat_i)`, together with a visual damage map and a count/area summary.

The difficult part is that damage is often defined by a change from the pre-disaster state. A post-only model may confuse pre-existing appearance, shadows, roof styles, vegetation, or image quality with damage. The paired image provides a direct reference, but it also introduces alignment, preprocessing, memory, and model-design issues. These trade-offs make the comparison meaningful even if the more complex model does not always win.

### 2.2 Research questions and hypotheses

**RQ1 - Does temporal context help?**  
Does a model using both `I_pre` and `I_post` obtain higher building-level Macro F1 than a post-only model on the same buildings, crops, split, and training budget?

**H1:** Paired imagery will improve Macro F1 and recall for `major` and `destroyed` buildings, although the gain may be smaller for visually ambiguous cases or strongly misaligned pairs.

**RQ2 - Does the fusion strategy matter?**  
Is explicit two-stream feature fusion more effective than simply concatenating the two images at the input?

**H2:** A two-stream model will provide a more stable comparison of the two dates because each image can be encoded separately before fusion. It may improve accuracy, but it will also require more memory and inference time.

**RQ3 - How much of the result comes from the localization stage?**  
How does damage classification change when the same classifier is evaluated on ground-truth building crops versus crops generated from predicted building masks?

**H3:** Predicted masks will reduce end-to-end performance relative to ground-truth crops, especially for small, partially visible, or heavily damaged buildings. Reporting both settings will distinguish classification quality from error propagation.

**RQ4 - Is the improvement robust across conditions?**  
Does the benefit of paired imagery remain consistent across disaster types and damage classes, or is it concentrated in a subset of events?

The project will treat these as testable questions rather than assumptions. A result in which paired imagery does not improve the main metric is still valid if the comparison is controlled and the failure is analyzed.

## 3. Planned readings and starting point

The reading plan follows the course guidance: first identify the task, main idea, and principal result from the abstract, figures, and conclusion; then inspect methods, experiments, implementation details, data, comparisons, and computing requirements for the papers that are most relevant.

| Reading | Role in this project | What we will extract |
|---|---|---|
| Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery* | Dataset and task definition | Pair structure, polygon labels, damage categories, disaster-event metadata, and known data limitations |
| Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* | Starting point for Stage A | Encoder-decoder design, skip connections, and a practical binary segmentation baseline |
| Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* | Optional Stage A comparison | Whether a stronger segmentation baseline is affordable and useful for the project scope |
| He et al., *Deep Residual Learning for Image Recognition* | Starting point for Stage B encoder | A standard, lightweight CNN backbone and the role of pretrained initialization |
| xView2/xBD official documentation and baseline resources | Implementation and data access | Download/access procedure, label format, baseline conventions, and practical preprocessing details |
| Course lecture L05, project guidance slides 64-73 | Project design constraints | Proposal content, credible baselines, reproducibility, feasibility checks, and report expectations |

The first implementation target is a small, complete run of an existing U-Net/classification implementation. Running this baseline will establish a real starting point and provide an early estimate of memory, runtime, and data-loading problems before we commit to the full model comparison.

## 4. Dataset and data protocol

### 4.1 Dataset

We plan to use the xBD dataset released through the xView2 project. It contains paired pre-disaster and post-disaster satellite imagery, building polygons, four-level damage labels, and disaster-event metadata. The full dataset is too large for an initial student project, so the first experiment will use a documented subset of approximately 3-5 disaster events. We will expand the subset only after the complete pilot run is stable.

Data access is a feasibility checkpoint, not an assumption. Before implementing the full models, the team will:

1. verify that the selected xBD data and labels can be downloaded or otherwise accessed;
2. record the dataset source, retrieval date, selected events, and any checksums or version identifiers available;
3. inspect actual pre/post examples, polygon overlays, label frequencies, image sizes, and alignment;
4. confirm that every selected sample has the files and labels needed by the planned pipeline; and
5. save a small data manifest so that the same subset can be reconstructed.

If the official data access is unavailable, the team will not silently replace it with an unrelated dataset. We will either use an accessible official subset or document a reduced fallback experiment based on the data that can be verified.

### 4.2 Splits and leakage prevention

The main split will be event-aware. Images and buildings from the same disaster event will not be spread across training, validation, and test sets. The initial subset will be arranged so that at least one event is held out for final testing when the number of available events permits it. If only three events can be used, we will use grouped splits and explicitly report the resulting limitation.

The split definition, random seeds, class counts, and preprocessing configuration will be fixed before the final training runs. Test labels will not be used for model selection, threshold tuning, augmentation decisions, or repeated debugging.

### 4.3 Input construction

For the controlled Stage B benchmark, a building polygon will be converted into a bounding-box crop with a fixed amount of surrounding context. A candidate crop size such as 128 x 128 pixels will be tested during the pilot; the final size and context margin will then be fixed and recorded. The same crop coordinates, resize rule, normalization, and augmentation policy will be used for every Stage B variant.

Paired augmentations must preserve the relationship between dates. Spatial transforms such as rotations and flips will be applied jointly to the pre/post pair. Photometric transforms will be conservative and will not create an artificial difference between dates. Invalid pairs, empty crops, unreadable images, and polygons outside the usable image area will be counted and documented rather than silently discarded.

For the end-to-end setting, Stage A will produce a predicted building mask. The mask will be converted into building instances or candidate connected components, and the same crop-and-classify procedure will be applied. The ground-truth-crop and predicted-mask settings will be evaluated separately.

### 4.4 Class imbalance and label quality

The damage labels are expected to be imbalanced, with fewer `major` and `destroyed` examples than `no damage`. The default training choice will be class-weighted cross-entropy, with weights computed from the training split only. Weighted sampling or focal loss may be used as a controlled ablation if the pilot shows that the default cannot learn minority classes; it will not be introduced after inspecting test results.

We will report class counts and inspect examples from all four classes. Possible label ambiguity, cloud/shadow interference, image misalignment, and small-building effects will be discussed in the error analysis.

## 5. Proposed system and technical contribution

### 5.1 Overall pipeline

```text
Pre-disaster image  ------------------------------┐
                                                  │
Post-disaster image --> Stage A building mask --> │
                                                  v
                         paired building crops --> Stage B damage classifier
                                                        |
                                                        v
                                      damage map + class counts + area summary
```

The controlled Stage B benchmark can bypass Stage A by using the ground-truth polygons:

```text
Ground-truth polygon --> identical paired crops --> B1 / B2 / B3 --> damage class
```

This separation is important because the main research question concerns the value of pre-disaster context, while a full pipeline also measures the separate difficulty of localizing buildings.

### 5.2 Stage A - building localization

The first localization model will be a U-Net-style binary segmentation network, using a lightweight CNN encoder. It will take the post-disaster image as input and predict a building mask. If resources permit, a DeepLabV3+ style model will be used as a secondary comparison, not as a requirement for project completion.

The localization loss will combine a pixel-wise term with a region-overlap term, for example binary cross-entropy plus Dice loss. The exact loss, input size, augmentation, and encoder initialization will be fixed after the pilot and applied consistently to the selected comparison. The output will be evaluated using IoU and Dice/F1, with precision and recall as supporting diagnostics.

### 5.3 Stage B - damage classification models

All Stage B variants will use the same crop size, split, label mapping, optimizer family, training budget, checkpoint rule, and evaluation code. The purpose is to compare input information and fusion design rather than to give one model an unfair tuning advantage.

**B0 - simple reference:** a majority-class or class-frequency reference, reported only to contextualize the difficulty of the task. It is not considered a competitive baseline.

**B1 - post-only baseline:** a CNN receives `I_post` and predicts one of the four damage classes. This is the main credible baseline because it uses the information available after the disaster without pre-disaster context.

**B2 - early-fusion baseline:** the pre/post channels are concatenated at the input and passed to a CNN. This tests whether simply providing both dates is enough.

**B3 - two-stream model:** the pre-disaster and post-disaster crops pass through two CNN branches, followed by feature fusion and a four-class prediction head. The first version will use a shared or matched backbone so that capacity remains comparable with B1 and B2. Fusion will occur after spatial feature extraction; the selected fusion point will be justified and fixed before the final test.

The initial backbone will be a compact ResNet-style encoder or equivalent CNN. ImageNet-pretrained initialization may be used if it is available in the execution environment; the initialization choice will be recorded and applied consistently across the comparison. If pretrained weights cannot be obtained, all models will use the same scratch initialization protocol.

### 5.4 Technical contribution

The technical contribution is an adaptation and empirical evaluation of established methods for a paired satellite-image damage-assessment task. Specifically, the project will contribute:

- a documented subset and event-aware split protocol for xBD;
- a reproducible polygon-to-mask and paired-crop pipeline;
- controlled post-only, early-fusion, and two-stream comparisons;
- an analysis separating building localization errors from damage-classification errors;
- per-class and per-disaster-type error analysis; and
- an end-to-end visualization that translates predictions into a damage map and area/count summary.

A new architecture, a state-of-the-art score, or a novel training algorithm is not required. The value of the project is a careful application, meaningful comparison, and evidence-based interpretation.

## 6. Experimental design and evaluation

### 6.1 Reproducibility controls

Before the final runs, the team will fix and record:

- dataset subset/version and event list;
- train/validation/test split manifest;
- crop size, context margin, resize rule, normalization, and augmentations;
- label mapping and class-weight computation;
- model architecture and parameter count;
- optimizer, learning-rate schedule, batch size, number of epochs, and regularization;
- random seeds and software/hardware environment; and
- the primary metric and checkpoint-selection rule.

Each model will be evaluated in the correct inference/evaluation mode. Training metrics will not be used as a substitute for validation results. The checkpoint with the best validation Macro F1 for Stage B, or the selected validation Dice/IoU rule for Stage A, will be restored before the single final test evaluation.

### 6.2 Metrics

**Stage A - building localization**

- Intersection over Union (IoU);
- Dice/F1 score;
- precision and recall; and
- qualitative overlays for representative easy and difficult scenes.

**Stage B - damage classification**

- **primary:** Macro F1, because the four classes are imbalanced;
- secondary: accuracy and balanced accuracy;
- per-class precision, recall, and F1;
- confusion matrix; and
- optional calibration or confidence summaries if the implementation is stable.

Accuracy alone will not be used to claim success. A model that predicts the majority class can have deceptively high accuracy while failing on `major` and `destroyed` buildings.

**Efficiency and operational cost**

For each main classifier, we will record parameter count, model size, peak memory where available, inference latency on a fixed hardware setup, and throughput. Runtime will be measured after warm-up on a fixed number of samples with the same batch size. If a model is described as more efficient, the claim will include both prediction quality and the measured cost.

### 6.3 Planned comparisons

The primary comparison is B3 versus B1 on the same held-out buildings and with the same evaluation protocol. The following comparisons will help distinguish plausible explanations:

| Comparison | Question it answers |
|---|---|
| B1 versus B2 | Does adding the pre-disaster image help when no explicit two-stream representation is used? |
| B2 versus B3 | Does the fusion design matter beyond the availability of the second image? |
| B1/B2/B3 on ground-truth crops | What can the classifier do when localization is not a confounder? |
| Best classifier on predicted-mask crops | How much performance is lost in the realistic end-to-end pipeline? |
| Pre/post image quality and alignment checks | Is a failure caused by the model or by information lost before the model receives it? |
| Disaster-type and class subgroups | Is the result robust or driven by a particular event/class? |
| Prediction quality versus latency/memory | Is any gain worth the additional computational cost? |

The last two comparisons follow the course guidance on understanding results. If paired imagery helps, we will examine whether the improvement comes from preserving information about change. If it does not help, we will inspect the pipeline for discarded context, crop misalignment, normalization problems, and localization errors before concluding that temporal information is unhelpful.

### 6.4 Repeated runs and uncertainty

If the compute budget permits, the final comparison will be repeated with at least three fixed random seeds and reported as mean plus standard deviation. If only one seed is feasible, the limitation will be stated and the run configuration will be made fully reproducible. Subgroup results will include sample counts so that very small groups are not over-interpreted.

## 7. Feasibility, risks, and fallback scope

| Risk | Why it matters | Mitigation and decision rule |
|---|---|---|
| Dataset access or download failure | The entire pipeline depends on paired images and labels | Verify access and inspect real examples first; record a manifest; use a documented reduced subset if necessary |
| Too many images or large files | Storage, preprocessing, and training may exceed the available budget | Start with 3-5 events, cache only the selected subset, and expand only after a complete pilot run |
| Stage A and Stage B are too much for one project | An end-to-end system can leave too little time for meaningful comparisons | Make ground-truth-crop B1/B2/B3 the minimum deliverable; add predicted-mask inference as the next tier |
| Strong class imbalance | Overall accuracy can hide failure on severe damage | Use training-only class weights, Macro F1, per-class metrics, and confusion matrices |
| Event or geographic leakage | Random image splits can make the test unrealistically easy | Split by disaster event and keep the final test event untouched |
| Pre/post misalignment or different image quality | The model may learn registration artifacts or fail to compare the dates | Inspect overlays, use joint spatial transforms, quantify invalid pairs, and include alignment-related error analysis |
| Two-stage error propagation | Poor localization can look like poor damage classification | Report ground-truth-crop and predicted-mask results separately |
| GPU memory or queue delays | The planned model comparison may not finish | Run a small complete model first, measure memory/runtime, use mixed precision or smaller crops if safe, checkpoint frequently, and reserve time for analysis |
| Label ambiguity and small objects | Some mistakes cannot be resolved from the image alone | Report qualitative examples and avoid over-interpreting individual predictions |

### Scope ladder

The project will use the following priority order:

1. **Must have:** verified data access, event-aware split, paired building-crop loader, B1 post-only baseline, B3 two-stream model, Macro F1/per-class evaluation, and an evidence-based comparison.
2. **Should have:** B2 early fusion, three-seed comparison, runtime/memory measurements, and subgroup/error analysis.
3. **Stretch:** U-Net Stage A, predicted-mask crops, end-to-end damage map, and a DeepLabV3+ comparison.

This ordering reserves time for comparisons and interpretation, rather than allowing the core project to be displaced by an over-ambitious pipeline.

## 8. Milestones and next steps

### Immediate next steps

1. Verify xBD/xView2 access and inspect actual paired samples and labels.
2. Create the event list, manifest, class-frequency table, and grouped train/validation/test split.
3. Implement polygon-to-mask conversion and a ground-truth building-crop dataloader.
4. Run one small, complete B1 training and evaluation cycle to estimate runtime, memory, and data-loader reliability.
5. Freeze the first preprocessing and evaluation configuration before expanding experiments.

### Planned milestones

| Phase | Main output | Completion evidence |
|---|---|---|
| Phase 1 - data audit | Verified subset and manifest | Real pre/post examples, polygon overlays, class counts, and split file |
| Phase 2 - controlled baseline | B1 post-only classifier | Reproducible validation/test script and baseline metrics |
| Phase 3 - temporal comparison | B2 and B3 results | Same split, same crop protocol, fixed seeds, and paired comparison table |
| Phase 4 - localization | U-Net Stage A | IoU/Dice results and qualitative mask overlays |
| Phase 5 - integration | Predicted-mask classifier and damage map | End-to-end example, counts/area summary, and error cases |
| Phase 6 - analysis and reporting | Final figures and report | Metric tables, efficiency measurements, limitations, contributions, and references |

The team will keep a short experiment log containing configuration, seed, checkpoint, validation score, test score, runtime, memory, and notes. This prevents the final report from relying on an untraceable best run.

## 9. Team roles and contributions

The final report will list all member names and describe actual contributions. The current responsibility split is:

| Member | Primary responsibility | Shared/secondary responsibility |
|---|---|---|
| Member A - Data and localization | Data access and parsing, polygon-to-mask conversion, event-aware splits, U-Net training, localization metrics | Data audit and experiment review |
| Member B - Damage classification | Ground-truth building crops, B1 post-only baseline, class-imbalance handling, training pipeline | Classification experiments and result checking |
| Member C - Paired-image modeling | B2 early fusion, B3 two-stream model, fusion comparison, disaster-type analysis | Runtime and ablation measurements |
| Member D - Evaluation and integration | Metric scripts, confusion matrices, visualizations, predicted-mask integration, damage-map demo | Error analysis, reproducibility log, and report figures |

All members will participate in scope decisions, experiment review, interpretation, and writing. The role table describes primary ownership, not separate reports.

## 10. Planned deliverables

- A documented xBD subset and event-aware split manifest, without committing the raw dataset to the repository.
- Reproducible preprocessing and paired-crop utilities.
- Stage B post-only, early-fusion, and two-stream experiments.
- Stage A localization results if the stretch scope is feasible.
- Evaluation tables with Macro F1, per-class metrics, confusion matrices, and subgroup results.
- Runtime/memory measurements under a fixed hardware setup.
- Qualitative overlays, damage maps, and an area/count summary.
- A final report using a conference-paper style template, with captions, citations, references, team member names, and contributions.

## 11. References

1. R. Gupta et al. "xBD: A Dataset for Assessing Building Damage from Satellite Imagery." arXiv:1911.09296, 2019. https://arxiv.org/abs/1911.09296
2. O. Ronneberger, P. Fischer, and T. Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015. https://arxiv.org/abs/1505.04597
3. L.-C. Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV, 2018. https://arxiv.org/abs/1802.02611
4. K. He et al. "Deep Residual Learning for Image Recognition." CVPR, 2016. https://arxiv.org/abs/1512.03385
5. xView2 project website and data resources. https://xview2.org/
6. DDA4220 course lecture L05, project guidance slides 64-73.

