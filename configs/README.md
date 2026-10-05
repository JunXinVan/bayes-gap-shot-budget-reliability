# Run configurations

One YAML per archived run. All configs are executed through the same entry point:

```bash
python -m overprompting_exp.cli {e1|e2} --config configs/<name>.yaml
```

`e1` logs risk/accuracy/entropy/disagreement curves over the shot grid; `e2` evaluates the offline selection rules on the logged traces. `run_e1_smoke.yaml` / `run_e2_smoke.yaml` are dummy-model fixtures for the pipeline smoke test (no GPU or LLM required).

## Naming

`<exp>_<model>_<dataset>_<family>_<protocol>[_variant].yaml`

- **exp**: `e1` (curves), `e2` (selection rules)
- **model**: `qwen2p5_3b` = Qwen2.5-3B-Instruct, `olmo3_7b_instruct` = OLMo-3-7B-Instruct
- **dataset**: `arc_1k` / `arc_8k` = ARC-Challenge, ManyICLBench 1k/8k prompt variants; `manyiclbench_arc_easy`, `manyiclbench_bbh_salient` = ARC-Easy / BBH-Salient via ManyICLBench; `hellaswag` = HellaSwag (official data), `hellaswag_1k` = HellaSwag via ManyICLBench
- **family**: `order`, `pad`, `template` prompt-perturbation families
- **protocol**: `full200` = fixed 200-query evaluation subset; `fast20` = small accuracy protocol; `fullval` = full HellaSwag validation split
- **variants**: `mainpert` = perturbation-averaged evaluation (mean over the family, excluding the base prompt); `seedN` = ManyICLBench split or replicate seed; `k32` = grid extended to k=32 (`baseonly`: k=32 endpoint only); `position_lock_v2` = query-position lock intervention; `shiftamp128` = delimiter-padding shift amplification; `separator` = separator-only control; `explicit*` = prompt-clarification controls; `densek` = dense k-grid

## Mapping to the manuscript

| Manuscript item | Configs |
|---|---|
| Main ARC curves (Figs. 2–4): Qwen2.5-3B, OLMo-3-7B × 1k/8k × Order/Pad/Template | `e1_*_arc_{1k,8k}_{order,pad,template}_full200[_mainpert]` |
| Split robustness (Suppl. Table: arc_split_robustness) | `*_mainpert_seed{1,2}` (Qwen, 1k/8k, all families) |
| Order ensembling (Table: order_ensemble) | `*_order_full200_mainpert` |
| Position lock (Table: poslock_arc_pad_ci) | `*_arc_8k_pad_full200_position_lock_v2[_seed{1,2}]` |
| Shift amplification / separator stress tests (Suppl. app:shiftamp) | `*_arc_8k_order_full200_shiftamp128`, `*_arc_8k_separator_full200` |
| Prompt-clarification control (Suppl. app:explicit_control) | `*_arc_8k_order_full200_explicit*` |
| Archived k=32 endpoints (Suppl. app:arc_protocol note) | `*_k32`, `*_k32_baseonly` |
| Multi-dataset accuracy summary (Table: multidataset_compact) | `e1_*_{hellaswag,manyiclbench_arc_easy,manyiclbench_bbh_salient}_order_fast20*` |
| HellaSwag held-out calibration audit (Fig. 5; Table: hellaswag_gated_candidates) | `e1_*_hellaswag_order_fullval_*` (post hoc calibration on the cached candidate distributions) |
| Earlier HellaSwag logged-curve check (Suppl. Table: hellaswag_full200_crossmodel) | `e1_*_hellaswag_order_full200_seed{42,456,789}`, `hellaswag_3b_reproduce_*`, `e1_qwen2p5_3b_hellaswag_1k_*` |
| Offline selection-rule sensitivity (Suppl. Table: lambda_sweep) | `e2_*` |

## Companion model family

Mistral-7B-Instruct-v0.3 appears in the manuscript as a companion family (Tables: multidataset_compact, order_ensemble_main, arc_split_robustness, and the HellaSwag audit). Those runs used the same protocols as the two primary families above with the model field swapped; the per-run configs for the companion family are not part of this archive.
