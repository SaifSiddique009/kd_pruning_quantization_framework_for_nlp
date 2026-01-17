# Phase 3: Comprehensive KD/Pruning Experiments

## Research Context

### Phase 1-2 Summary
- **Phase 1**: Full finetuning of large models (XLM-RoBERTa, BanglaBERT) → Teacher models
- **Phase 2**: Full finetuning of small models (SahajBERT, BanglaBERT-small) → Finetuned students
- **Phase 3** (This): Knowledge Distillation + Pruning experiments

### Important Notes
- Phase 1-2 used **F1 Weighted** by mistake
- Phase 3 uses **F1 Macro** as the primary metric
- All experiments use `--eval_fold 3` with `--use_original_folds` for consistent evaluation

### Pruning Methods & Parameter Reduction

| Method | Type | Reduces Parameters? | Speedup on Kaggle? |
|--------|------|---------------------|-------------------|
| **Magnitude** | Unstructured | NO (zeros weights) | NO |
| **Wanda** | Unstructured | NO (zeros weights) | NO |
| **Gradual** | Unstructured | NO (zeros weights) | NO |
| **Structured** | Structured | **YES** (removes heads) | **YES** |

**Key insight**: Only structured pruning actually reduces model size and gives real speedup on standard hardware!

---

## Models Registry

### Teacher Models (Large, Finetuned in Phase 1)

| ID | Model | HuggingFace Path |
|----|-------|------------------|
| **T1** | XLM-RoBERTa | `Saif-Siddique/bangla-cyberbully-xlm-roberta-base` |
| **T2** | BanglaBERT | `Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base` |

### Finetuned Student Models (Small, Finetuned in Phase 2)

| ID | Model | HuggingFace Path |
|----|-------|------------------|
| **FS1** | SahajBERT-ft | `Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT` |
| **FS2** | BanglaBERT-small-ft | `Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small` |

### Raw Student Models (Not Finetuned)

| ID | Model | HuggingFace Path |
|----|-------|------------------|
| **RS1** | SahajBERT-raw | `neuropark/sahajBERT` |
| **RS2** | BanglaBERT-small-raw | `csebuetnlp/banglabert_small` |

---

## KD Combinations (4 Total)

All teacher-student combinations for comprehensive comparison:

| KD ID | Teacher | Raw Student | Output Name |
|-------|---------|-------------|-------------|
| **KD1** | T1 (XLM-RoBERTa) | RS1 (SahajBERT) | `kd_T1_RS1` |
| **KD2** | T1 (XLM-RoBERTa) | RS2 (BanglaBERT-small) | `kd_T1_RS2` |
| **KD3** | T2 (BanglaBERT) | RS1 (SahajBERT) | `kd_T2_RS1` |
| **KD4** | T2 (BanglaBERT) | RS2 (BanglaBERT-small) | `kd_T2_RS2` |

---

## Experiment Summary

| Scenario | Description | Runs |
|----------|-------------|------|
| **1** | Baselines (T1, T2, FS1, FS2) | 4 |
| **2** | KD Only (KD1, KD2, KD3, KD4) | 4 |
| **3** | Prune with Magnitude (FS1, FS2, KD1-4) | 6 |
| **4** | Pruning Methods (wanda, gradual, structured on KD1-4) | 12 |
| **Total** | | **26** |

---

## Optimized Execution Strategy

### Key Optimization: Reuse KD Models!

Instead of redoing KD for each pruning experiment, we:
1. **Scenario 2**: Run KD once per combination, SAVE the distilled models
2. **Scenario 3 & 4**: Load saved KD models, apply different pruning methods

This saves ~4 hours per pruning experiment!

### Execution Order

```
PHASE 1: Independent Runs (Parallel)
├── Scenario 1: All baselines (4 runs)
└── Scenario 2: All KD experiments (4 runs) → SAVE MODELS!

PHASE 2: Depends on Scenario 2 (Parallel after PHASE 1)
├── Scenario 3: Prune FS1, FS2 (2 runs) - can start immediately
├── Scenario 3: Prune KD1-4 with magnitude (4 runs)
└── Scenario 4: Prune KD1-4 with wanda, gradual, structured (12 runs)
```

### Kaggle Account Distribution

| Account | Runs | Est. Time | Notes |
|---------|------|-----------|-------|
| **1** | 1.1-1.4 (baselines) | ~2h | Independent |
| **2** | 2.1-2.4 (KD) | ~6h | **CRITICAL** - Save models! |
| **3** | 3.1-3.2 (prune finetuned) | ~2h | Independent |
| **4** | 3.3-3.6 (prune KD, magnitude) | ~4h | Wait for Account 2 |
| **5** | 4.1-4.12 (other methods) | ~8h | Wait for Account 2 |

---

## Default Hyperparameters

### Knowledge Distillation
| Parameter | Value | CLI Flag |
|-----------|-------|----------|
| Epochs | 15 | `--epochs` |
| Batch Size | 32 | `--batch` |
| Learning Rate | 2e-5 | `--lr` |
| Alpha | 0.7 | `--kd_alpha` |
| Temperature | 4.0 | `--kd_temperature` |

### Pruning
| Parameter | Value | CLI Flag |
|-----------|-------|----------|
| Target Sparsity | 50% | `--prune_sparsity` |
| Fine-tune Epochs | 3 | `--fine_tune_epochs` |
| Fine-tune After | True | `--fine_tune_after_prune` |

---

## Scenario 1: Baselines (4 runs)

### Run 1.1: Teacher T1 - XLM-RoBERTa
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario1/T1_baseline
```

### Run 1.2: Teacher T2 - BanglaBERT
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario1/T2_baseline
```

### Run 1.3: Finetuned Student FS1 - SahajBERT
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario1/FS1_baseline
```

### Run 1.4: Finetuned Student FS2 - BanglaBERT-small
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario1/FS2_baseline
```

---

## Scenario 2: KD Only (4 runs) - SAVE THESE MODELS!

**IMPORTANT**: Save the output models from these runs. They will be reused in Scenarios 3 and 4!

### Run 2.1: KD1 - T1 -> RS1 (XLM-RoBERTa -> SahajBERT)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario2/KD1_T1_RS1

# SAVE MODEL: Copy ./results/scenario2/KD1_T1_RS1/model_final_hf to reuse!
```

### Run 2.2: KD2 - T1 -> RS2 (XLM-RoBERTa -> BanglaBERT-small)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario2/KD2_T1_RS2
```

### Run 2.3: KD3 - T2 -> RS1 (BanglaBERT -> SahajBERT)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "neuropark/sahajBERT" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario2/KD3_T2_RS1
```

### Run 2.4: KD4 - T2 -> RS2 (BanglaBERT -> BanglaBERT-small)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario2/KD4_T2_RS2
```

---

## Scenario 3: Pruning with Magnitude (6 runs)

Prune finetuned students and KD models using magnitude pruning.

### Run 3.1: Prune FS1 (Finetuned SahajBERT)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline prune_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/FS1_magnitude
```

### Run 3.2: Prune FS2 (Finetuned BanglaBERT-small)
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline prune_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/FS2_magnitude
```

### Run 3.3: Prune KD1 with Magnitude
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/KD1_magnitude
```

### Run 3.4: Prune KD2 with Magnitude
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/KD2_magnitude
```

### Run 3.5: Prune KD3 with Magnitude
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/KD3_magnitude
```

### Run 3.6: Prune KD4 with Magnitude
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method magnitude --prune_sparsity 0.5 \
    --fine_tune_after_prune --fine_tune_epochs 3 \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario3/KD4_magnitude
```

---

## Scenario 4: Pruning Methods Comparison (12 runs)

Compare wanda, gradual, and structured pruning on all 4 KD models.
(Magnitude already done in Scenario 3)

### Wanda Pruning (4 runs)

#### Run 4.1: KD1 + Wanda
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method wanda --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD1_wanda
```

#### Run 4.2: KD2 + Wanda
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method wanda --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD2_wanda
```

#### Run 4.3: KD3 + Wanda
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method wanda --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD3_wanda
```

#### Run 4.4: KD4 + Wanda
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method wanda --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD4_wanda
```

### Gradual Pruning (4 runs)

#### Run 4.5: KD1 + Gradual
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method gradual --prune_sparsity 0.5 --prune_schedule cubic \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD1_gradual
```

#### Run 4.6: KD2 + Gradual
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method gradual --prune_sparsity 0.5 --prune_schedule cubic \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD2_gradual
```

#### Run 4.7: KD3 + Gradual
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method gradual --prune_sparsity 0.5 --prune_schedule cubic \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD3_gradual
```

#### Run 4.8: KD4 + Gradual
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method gradual --prune_sparsity 0.5 --prune_schedule cubic \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD4_gradual
```

### Structured Pruning (4 runs) - ACTUALLY REDUCES PARAMETERS!

#### Run 4.9: KD1 + Structured
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method structured --prune_sparsity 0.3 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD1_structured
```

#### Run 4.10: KD2 + Structured
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method structured --prune_sparsity 0.3 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD2_structured
```

#### Run 4.11: KD3 + Structured
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method structured --prune_sparsity 0.3 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD3_structured
```

#### Run 4.12: KD4 + Structured
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base" \
    --student_path "csebuetnlp/banglabert_small" \
    --prune_method structured --prune_sparsity 0.3 \
    --fine_tune_after_prune \
    --use_original_folds --eval_fold 3 \
    --output_dir ./results/scenario4/KD4_structured
```

---

## Results Directory Structure

```
./results/
├── scenario1/
│   ├── T1_baseline/
│   ├── T2_baseline/
│   ├── FS1_baseline/
│   └── FS2_baseline/
├── scenario2/
│   ├── KD1_T1_RS1/     # SAVE for reuse!
│   ├── KD2_T1_RS2/
│   ├── KD3_T2_RS1/
│   └── KD4_T2_RS2/
├── scenario3/
│   ├── FS1_magnitude/
│   ├── FS2_magnitude/
│   ├── KD1_magnitude/
│   ├── KD2_magnitude/
│   ├── KD3_magnitude/
│   └── KD4_magnitude/
└── scenario4/
    ├── KD1_wanda/
    ├── KD2_wanda/
    ├── KD3_wanda/
    ├── KD4_wanda/
    ├── KD1_gradual/
    ├── KD2_gradual/
    ├── KD3_gradual/
    ├── KD4_gradual/
    ├── KD1_structured/
    ├── KD2_structured/
    ├── KD3_structured/
    └── KD4_structured/
```

---

## Why Not Reuse KD in Current Implementation?

**Current limitation**: The `kd_prune` pipeline does KD fresh every time.

**Workaround if you want to reuse KD models**:
1. Run Scenario 2 first with `kd_only`
2. Upload saved models to HuggingFace
3. Use `prune_only` with saved KD model as `teacher_checkpoint`

```bash
# Example: After uploading KD1 to HuggingFace
!python main.py \
    --pipeline prune_only \
    --teacher_checkpoint "YOUR_USERNAME/kd1_sahajbert_distilled" \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --output_dir ./results/scenario4/KD1_wanda_reused
```

**For this experiment set**: We use `kd_prune` for simplicity, accepting the redundant KD computation. Future optimization could implement model caching.

---

## Quick Reference

### Model Paths
| ID | Path |
|----|------|
| T1 | `Saif-Siddique/bangla-cyberbully-xlm-roberta-base` |
| T2 | `Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base` |
| FS1 | `Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT` |
| FS2 | `Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small` |
| RS1 | `neuropark/sahajBERT` |
| RS2 | `csebuetnlp/banglabert_small` |

### Pruning Methods
| Method | CLI Flag | Reduces Params? |
|--------|----------|-----------------|
| Magnitude | `--prune_method magnitude` | No |
| Wanda | `--prune_method wanda` | No |
| Gradual | `--prune_method gradual --prune_schedule cubic` | No |
| Structured | `--prune_method structured` | **Yes** |

### Quick Test
```bash
!python main.py \
    --dataset_path "/kaggle/input/bangla-cyberbully-dataset/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "test" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --data_fraction 0.1 --epochs 2 \
    --output_dir ./results/quick_test
```
