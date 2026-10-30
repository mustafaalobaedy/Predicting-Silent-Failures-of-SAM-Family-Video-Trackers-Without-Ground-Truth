# Predicting Silent Failures of SAM-Family Video Trackers Without Ground Truth

Code, extracted signals and analysis scripts for the paper *"Predicting Silent Failures of SAM-Family Video Trackers Without Ground Truth: Detection, Calibrated Alarms, and Correction Scheduling"* (submitted to the *Journal of Imaging*).

**Author:** Mustafa Muwafak Alobaedy, Centre for Image and Vision Computing, Multimedia University, Cyberjaya, Malaysia (mustafa.alobaedy@mmu.edu.my)

## Overview
SAM 2 and its successors propagate a prompted object through a video, but their errors are silent: a wrong or empty mask is returned without warning. This repository contains a lightweight failure predictor that runs next to an unchanged tracker and needs no ground truth at test time.
- **Signals:** 14 signals are read from a single forward pass of the tracker:
  - the three candidate IoU predictions and the object score;
  - object-pointer drift;
  - mask-shape statistics.
- **Detector:** these signals, plus causal temporal features, feed a shallow gradient-boosted decision tree (GBDT) trained with episode-balanced weights.
- **Calibrated alarms:** the alarm threshold is calibrated on held-out videos by conformal risk control.
- **Correction scheduling:** the failure score is used to schedule corrective clicks.
- **Trackers:** SAM 2.1, EdgeTAM and the tracker of SAM 3.

Episodes of failure detected at a 5% false-alarm rate (τ = 0.5), MOSE and SA-V:

| Tracker | Tracker's predicted IoU | Proposed (episode-weighted GBDT) | Proposed, trained on SAM 2.1 only |
|---|---|---|---|
| SAM 2.1 | 9–14% | 64–85% | — |
| EdgeTAM | 4–7% | 70–89% | 68–88% |
| SAM 3 | 4–6% | 69–93% | 70–91% |

## Repository layout
| Path | Content |
|---|---|
| `extract_signals.py` | Runs a tracker (`--model sam2`, `edgetam` or `sam3`) on a dataset with mask or click prompts and writes one CSV row per object and frame: the signals, and the ground-truth IoU and presence used only for labels. Optional memory-attention statistics. |
| `analyze.py` | Frame-level evaluation: presence-aware labels, single signals, e-process baseline, LR and GBDT with 5-fold cross-validation grouped by video (AUROC, AUPRC, recall at 10% FPR). |
| `event_eval.py` | Event-level evaluation: failure episodes, detection rate at matched false-alarm rates; frame-level, onset-only and episode-weighted GBDTs. |
| `bootstrap_events.py` | Video-level bootstrap confidence intervals for the event-level results. |
| `gbdt_regularization.py` | GBDT configuration grid and nested cross-validation. |
| `conformal_alarm.py` | Alarm thresholds calibrated by conformal risk control, within and across datasets. |
| `click_schedule.py` | Simulated interactive correction: oracle, GBDT, predicted-IoU, uniform and hybrid schedulers. |
| `transfer_model.py` | Zero-shot transfer: GBDT trained on SAM 2.1 signals, applied to another tracker. |
| `vlm_gate.py` | Verification gate with Qwen2.5-VL (mask overlays, VLM scoring, analysis). |
| `fetch_mose_subset.py`, `make_sav_6fps.py`, `prep_lvos.py` | Data preparation for MOSE, SA-V and LVOS-v2. |
| `run_*.sh` | The run scripts used for the paper (resumable; long runs). |
| `paths.py` | Locations of data, checkpoints and third-party code (environment variables). |
| `results/` | Extracted signals for all settings (`*.csv.gz`) and every analysis output used in the paper. |
| `paper/` | `paper_numbers.py` and `journal_numbers.py` recompute every number in the paper's tables and figures from `results/`. |

## Setup
1. **Python 3.10 and packages.** Install PyTorch for your CUDA version first, then:
   ```bash
   pip install -r requirements.txt
   ```
2. **SAM 2.1.** Install SAM 2 from https://github.com/facebookresearch/sam2 (commit `2b90b9f`) with `pip install -e .`. Download `sam2.1_hiera_base_plus.pt` into `$SAMFAIL_CKPT`.
3. **EdgeTAM (optional).** Clone https://github.com/facebookresearch/EdgeTAM (commit `7711e01`) into `$SAMFAIL_EXT/EdgeTAM`, with its checkpoint in `$SAMFAIL_EXT/EdgeTAM/checkpoints/edgetam.pt`.
   - With torch ≥ 2.11, replace `.expand(...).view(...)` by `.reshape(...)` in `sam2/modeling/perceiver.py`. The values are identical.
4. **SAM 3 (optional).** Clone https://github.com/facebookresearch/sam3 (commit `f1647e8`) into `$SAMFAIL_EXT/sam3`. Download the gated checkpoint `sam3.pt` (Hugging Face `facebook/sam3`, access on request) into `$SAMFAIL_CKPT`.
5. **Qwen2.5-VL (optional, VLM gate).** Download `Qwen/Qwen2.5-VL-3B-Instruct` into `$SAMFAIL_EXT/models/Qwen2.5-VL-3B-Instruct`.
6. **Locations.** Set the locations, or keep the defaults relative to the repository:
   ```bash
   export SAMFAIL_DATA=/path/to/datasets   # DAVIS2017/, MOSE/, SA-V/, LVOS2/
   export SAMFAIL_CKPT=/path/to/checkpoints
   export SAMFAIL_EXT=/path/to/third_party
   export SAMFAIL_CACHE=/path/to/cache     # VLM-gate overlays
   ```

## Data
| Dataset | Use | Preparation |
|---|---|---|
| DAVIS 2017 val (30 videos) | development | official release under `$SAMFAIL_DATA/DAVIS2017/DAVIS` |
| MOSE (200 training videos, seed 0) | main | `python fetch_mose_subset.py --n 200 --seed 0 --root $SAMFAIL_DATA/MOSE` (streams the archive from Hugging Face `FudanCVL/MOSE`) |
| SA-V val (155 videos) | main | official release (license form at ai.meta.com) under `$SAMFAIL_DATA/SA-V/sav_val`; then `python make_sav_6fps.py` (tracking on the annotated 6-fps frames) |
| LVOS-v2 val (139 of 140 videos) | long videos | official release under `$SAMFAIL_DATA/LVOS2/valid`; then `python prep_lvos.py` |

## Reproducing the results
**Precomputed signals.** The extracted signals are included, so the analyses can be run without a GPU:
```bash
gunzip -k results/*.csv.gz
python analyze.py results/mose_mask.csv results/mose_click.csv results/sav_mask.csv results/sav_click.csv
python event_eval.py results/mose_mask.csv results/mose_click.csv results/sav_mask.csv results/sav_click.csv
python bootstrap_events.py results/mose_mask.csv results/mose_click.csv results/sav_mask.csv results/sav_click.csv
python conformal_alarm.py results/mose_mask.csv results/mose_click.csv results/sav_mask.csv results/sav_click.csv --R 100   # ~4 h CPU
python transfer_model.py results/mose_mask.csv:results/edgetam_mose_mask.csv results/mose_click.csv:results/edgetam_mose_click.csv \
    results/sav_mask.csv:results/edgetam_sav_mask.csv results/sav_click.csv:results/edgetam_sav_click.csv --tag edgetam
python paper/paper_numbers.py /tmp && python paper/journal_numbers.py   # every number of the paper -> paper/journal_numbers.json
```
- **LVOS-v2, trackers and attention ablation:** the same scripts take `--tag lvos`, `--tag sam3` and `--tag attn` (see `run_lvos.sh`, `run_edgetam_analysis.sh` and `run_sam3.sh`).

**Signal extraction (GPU).** Re-extracting the signals takes about 1.5–3.5 h per setting on a Jetson AGX Orin:
```bash
python extract_signals.py --dataset davis --prompt mask --out results/davis_mask.csv --limit 2   # smoke test
python extract_signals.py --dataset mose --prompt mask --out results/mose_mask.csv
python extract_signals.py --model edgetam --dataset sav --prompt click --lazy --no-attn --out results/edgetam_sav_click.csv
python extract_signals.py --model sam3 --dataset mose --prompt mask --lazy --no-attn --out results/sam3_mose_mask.csv
```
- **Resuming:** extraction is resumable, so videos already present in `--out` are skipped.
- **Long videos:** `--lazy` decodes frames on demand and is needed for LVOS-v2.
- **Attention statistics:** `--no-attn` skips the memory-attention statistics, which double the run time.
- **Interactive correction and VLM gate:**
  - correction scheduling: `run_clicks.sh` and `run_clicks_hybrid.sh`;
  - VLM gate: `run_vlm_gate.sh`, then `python vlm_gate.py analyze`.

## Notes
- **Labels:** a frame fails if the target is visible and IoU < τ, or if the target is absent and the mask is non-empty. τ ∈ {0.5, 0.7}.
- **Ground truth:** the ground-truth columns of the signal CSVs are used only for labels, never as features.
- **e-process baseline:** the e-process baseline is a windowed re-implementation of Monroy Muñoz et al. (arXiv:2602.12983), not the published code.
- **Frame rates:** SA-V and LVOS-v2 are tracked at 6 fps (their annotated frames), not at their full frame rates.
- **Reproducibility:** all random choices use seed 0. The cross-validation is grouped by video, and no video contributes frames to both training and testing.

## License
The code is released under the MIT License (see `LICENSE`). The datasets keep their own licenses: DAVIS 2017, MOSE, SA-V and LVOS-v2 must be obtained from their providers. The extracted signals in `results/` are derived from them.
