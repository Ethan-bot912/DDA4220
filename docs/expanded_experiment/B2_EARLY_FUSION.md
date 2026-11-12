# Expanded B2 — primary early-fusion control

## Purpose

This is the primary, controlled B2 comparison for the strongest comparable
expanded B1 run: `configs/swin_t_expanded_b1_seed42_bs16.json`, whose validation pooled Macro
F1 is 0.5075485.  It tests whether access to a paired pre-disaster crop helps
when every other experimental choice is held fixed.

The B1 result on the separate FE-001 split is not used here because its
training and validation events differ from the expanded protocol.

## Frozen comparison contract

| Item | B1 reference | B2 primary |
| --- | --- | --- |
| Input | post RGB | pre RGB concatenated with post RGB |
| Input shape | 3 x 224 x 224 | 6 x 224 x 224 |
| Backbone | ImageNet-pretrained Swin-T | same Swin-T, only patch stem expanded |
| Crop, resize and context | original GeoTIFF, post-bbox window, 224 px, context 16 | identical paired window and resize |
| Normalization | ImageNet RGB | apply ImageNet RGB normalization independently to pre and post, then concatenate |
| Augmentation | none | none |
| Split, seed and training budget | frozen B1 configuration | identical |
| Checkpoint rule | pooled validation Macro F1, strict improvement, earlier tie | identical |

The B2 patch embedding changes from `Conv2d(3, 96, kernel_size=4, stride=4, bias=True)`
to `Conv2d(6, 96, kernel_size=4, stride=4, bias=True)`. Given its
ImageNet-pretrained RGB weight tensor `W`, initialize it as `concat(W, W) / 2`
and copy the original bias unchanged. All other weights start from the same
ImageNet-pretrained Swin-T base as B1. Do not warm-start
from the already fine-tuned B1 checkpoint unless a separate, matched B1
warm-start control is also run.

## Required preflight checks

1. Each paired row has readable pre/post files, matching crop shape and finite
   values in `[0, 1]` before normalization.
2. Pre and post use exactly the same raster window derived from the post
   building box.
3. With a synthetic `pre == post` input and the expanded-stem initialization,
   the B2 stem output matches the original B1 stem output to numerical
   tolerance.
4. The train/validation row counts, event lists and manifest hash match the
   B1 reference; no test images or labels are read.
5. Save the configuration, source hash, environment, manifest hash and B2
   checkpoint hash in a new output directory.

## Evaluation

The primary result is B2 minus B1 pooled validation Macro F1 on the same
45,002 buildings.  Also report event-level Macro F1, accuracy, balanced
accuracy, Macro-AP, every class's precision/recall/F1, confusion matrices,
and paired corrected/new-error counts.  Record parameters, peak memory,
training time, and both resident-tensor and end-to-end paired-input inference
costs.

The sealed Nepal, Florence and Santa Rosa events remain untouched until both
models and the evaluation protocol are frozen.

The completed run record is in
`reports/expanded_swin_t_b2_early_seed42_bs16/`; its best epoch is 8 and its
validation pooled Macro-F1 is 0.498054, below the B1 reference 0.507549.

## Scope boundary

This first run is early 6-channel fusion only.  Two-stream/late fusion,
explicit difference or ratio features, alignment correction, post-hoc
threshold tuning, and extra losses are follow-up ablations, not part of the
primary B1--B2 claim.
