# Experiment Scenarios: KD/Pruning/Quantization Comparison

## Models Overview

| Type | Model Path | Description |
|------|------------|-------------|
| Teacher | `Saif-Siddique/bangla-cyberbully-xlm-roberta-base` | XLM-RoBERTa based |
| Finetuned Student | `Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT` | SahajBERT finetuned |
| Raw Student | `neuropark/sahajBERT` | SahajBERT (not finetuned) |

---

## Default Hyperparameters

### Knowledge Distillation
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Epochs | 15 | `--epochs` |
| Batch Size | 32 | `--batch` |
| Learning Rate | 2e-5 | `--lr` |
| Alpha | 0.7 | `--kd_alpha` |
| Temperature | 4.0 | `--kd_temperature` |

### Pruning
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Target Sparsity | 50% | `--prune_sparsity` |
| Method | magnitude | `--prune_method` |
| Fine-tune Epochs | 3 | `--fine_tune_epochs` |

### Quantization
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Method | dynamic | `--quant_method` |
| Data Type | int8 | `--quant_dtype` |

---

## Output Directory Structure

```
./results/
├── scenario1/
│   ├── teacher_baseline/
│   └── finetuned_student_baseline/
├── scenario2/
│   └── kd_only/
├── scenario3/
│   ├── prune_finetuned/
│   └── prune_distilled/
├── scenario4/
│   ├── quant_finetuned/
│   └── quant_distilled/
├── scenario5/
│   ├── prune_quant_finetuned/
│   └── kd_prune_quant/
├── scenario6/
│   ├── magnitude/
│   ├── wanda/
│   └── gradual/
├── scenario7/
│   ├── fp16/
│   ├── int8/
│   └── int4/
└── scenario8/
    └── hp_a{alpha}_t{temp}_lr{lr}/
```

---

## Scenario 1: Baseline Performance Test (2 runs)

Evaluate Teacher and Finetuned Student models without any compression.

### Run 1.1: Teacher Baseline
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario1/teacher_baseline
```

### Run 1.2: Finetuned Student Baseline
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario1/finetuned_student_baseline
```

---

## Scenario 2: KD Only (1 run)

Apply Knowledge Distillation from Teacher to Raw Student.

### Run 2.1: KD from Teacher to Raw Student
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario2/kd_only
```

---

## Scenario 3: Prune Only (2 runs)

Prune Finetuned Student and Distilled Student, then fine-tune.

### Run 3.1: Prune Finetuned Student
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline prune_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario3/prune_finetuned
```

### Run 3.2: Prune Distilled Student (KD -> Prune)
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario3/prune_distilled
```

---

## Scenario 4: Quant Only (2 runs)

Quantize Finetuned Student and Distilled Student.

### Run 4.1: Quantize Finetuned Student
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline quant_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --quant_method dynamic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario4/quant_finetuned
```

### Run 4.2: Quantize Distilled Student (KD -> Quant)
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quant_method dynamic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario4/quant_distilled
```

---

## Scenario 5: Prune + Quant (2 runs)

Apply both pruning and quantization.

### Run 5.1: Prune + Quant Finetuned Student
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --quant_method dynamic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario5/prune_quant_finetuned
```

### Run 5.2: KD + Prune + Quant Distilled Student
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --quant_method dynamic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario5/kd_prune_quant
```

---

## Scenario 6: Pruning Methods Comparison (3 runs)

Compare magnitude, wanda, and gradual pruning on Distilled Student.

### Run 6.1: Magnitude Pruning
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario6/magnitude
```

### Run 6.2: Wanda Pruning
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario6/wanda
```

### Run 6.3: Gradual Pruning
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --prune_method gradual \
    --prune_sparsity 0.5 \
    --prune_schedule cubic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario6/gradual
```

---

## Scenario 7: Quantization Methods Comparison (3 runs)

Compare fp16, int8, and int4 quantization on Distilled Student.

### Run 7.1: FP16 Quantization
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quant_method fp16 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario7/fp16
```

### Run 7.2: INT8 Dynamic Quantization
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quant_method dynamic \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario7/int8
```

### Run 7.3: INT4 Quantization
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quant_method int4 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario7/int4
```

---

## Scenario 8: KD Hyperparameter Tuning (8 runs)

Grid search over alpha, temperature, and learning rate.

| Run | Alpha | Temperature | Learning Rate |
|-----|-------|-------------|---------------|
| 8.1 | 0.5 | 2.0 | 1e-5 |
| 8.2 | 0.5 | 2.0 | 5e-5 |
| 8.3 | 0.5 | 6.0 | 1e-5 |
| 8.4 | 0.5 | 6.0 | 5e-5 |
| 8.5 | 0.9 | 2.0 | 1e-5 |
| 8.6 | 0.9 | 2.0 | 5e-5 |
| 8.7 | 0.9 | 6.0 | 1e-5 |
| 8.8 | 0.9 | 6.0 | 5e-5 |

### Run 8.1: alpha=0.5, temp=2.0, lr=1e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.5 \
    --kd_temperature 2.0 \
    --lr 1e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a05_t2_lr1e5
```

### Run 8.2: alpha=0.5, temp=2.0, lr=5e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.5 \
    --kd_temperature 2.0 \
    --lr 5e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a05_t2_lr5e5
```

### Run 8.3: alpha=0.5, temp=6.0, lr=1e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.5 \
    --kd_temperature 6.0 \
    --lr 1e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a05_t6_lr1e5
```

### Run 8.4: alpha=0.5, temp=6.0, lr=5e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.5 \
    --kd_temperature 6.0 \
    --lr 5e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a05_t6_lr5e5
```

### Run 8.5: alpha=0.9, temp=2.0, lr=1e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.9 \
    --kd_temperature 2.0 \
    --lr 1e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a09_t2_lr1e5
```

### Run 8.6: alpha=0.9, temp=2.0, lr=5e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.9 \
    --kd_temperature 2.0 \
    --lr 5e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a09_t2_lr5e5
```

### Run 8.7: alpha=0.9, temp=6.0, lr=1e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.9 \
    --kd_temperature 6.0 \
    --lr 1e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a09_t6_lr1e5
```

### Run 8.8: alpha=0.9, temp=6.0, lr=5e-5
```bash
!python main.py \
    --dataset_path "/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv" \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.9 \
    --kd_temperature 6.0 \
    --lr 5e-5 \
    --use_original_folds \
    --eval_fold 3 \
    --output_dir ./results/scenario8/hp_a09_t6_lr5e5
```

---

## Summary

| Scenario | Description | Runs |
|----------|-------------|------|
| 1 | Baseline (Teacher + Finetuned Student) | 2 |
| 2 | KD Only | 1 |
| 3 | Prune Only | 2 |
| 4 | Quant Only | 2 |
| 5 | Prune + Quant | 2 |
| 6 | Pruning Methods (magnitude/wanda/gradual) | 3 |
| 7 | Quant Methods (fp16/int8/int4) | 3 |
| 8 | KD Hyperparameter Tuning | 8 |
| **Total** | | **23** |
