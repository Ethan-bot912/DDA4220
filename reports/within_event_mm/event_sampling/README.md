# Event sampling experiment card

ID: WE-003 (local provisional). Designed 2026-10-26 before running.
Status: synthetic acceptance passed; formal state is recorded in [RUN_STATUS.md](RUN_STATUS.md).
No scientific result until all four runs and paired export complete.

Question: does event balancing within each damage class improve severity judgments?
Plan: [prespecified design](../../../docs/within_event_mm/EVENT_SAMPLING_EXPERIMENT_PLAN.md).
Control: natural permutation; candidate: replacement sampling, balanced events within
unchanged per-class draw quotas. Both retain inverse-frequency loss weights.
Models: B1 post, B2 early. Seed42, attempt01, 20 epochs each, matched rebuilt A.
Data: train 26976 / val 9014, same frozen tile split and verified crop cache.
Input: 128px, context16, float32, workers4, batch32; no augmentation or pretrained weights.
Selection: best pooled val Macro F1, strict improvement; all event scores from that best.

Scientific scope: within-event development only, single seed, no significance or
stability claim. No test images, cross-event claim, GitHub push, PR or Release.
The original class-weight experiment and all historical source files stay intact.

Artifacts: full checkpoints/predictions/logs in output/within_event_mm/event_sampling/;
small results and run evidence in this report directory, by condition/model/seed/attempt.
All source/config/data hashes, initialization, RNG, per-epoch exposure and order hashes
are recorded. New runs never overwrite an existing attempt.

Judgment: report pooled, class and event deltas and their tradeoffs; no arbitrary
numeric pass threshold. Positive pooled F1 alone does not establish the mechanism.
All outcomes are retained; multi-seed confirmation is a later stage. See plan.

Evidence: [distribution diagnosis](distribution_diagnosis.json), [review](REVIEW.md), [software acceptance](verification.json).

Final pre-formal plan includes the user clarification about the common within-event benchmark.
Initial draft preceded implementation; final plan was expanded before formal launch.

## 已完成正式结果

2026-10-26 03:14四组及val配对检查完成；[中文结论](INTERPRETATION_ZH.md)、[结果](results.csv)、[恢复审计](recovery_audit.json)。两模型pooled F1均下降，本候选不作为后续基线。
