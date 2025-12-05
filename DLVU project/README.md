# Face Verification with Margin-Based Embedding
**EE40034 Deep Learning for Visual Understanding – Final Project**  
**Student:** Junsu Kim (20210151)

---

## 1. Overview

This repository contains the full implementation, training scripts, and test results for my face verification project.  
The experiments evaluate how **loss functions**, **optimizers**, **learning-rate schedulers**, and **backbone regularization** affect margin-based embedding learning.

The project follows a **two-stage training pipeline**:

- **Train1:** From-scratch embedding learning  
- **Train2:** Fine-tuning using the best Train1 checkpoint  

The final model (**Exp04-AAM**) achieves:

- **Validation EER:** 8.51%  
- **Test EER:** 9.219% (official evaluation)

The required test output file is included as:

`exp04-AAM_test_ep12.csv`

---

## 2. File Structure

```text
trainEmbedNet.py         Main training & evaluation script  
DatasetLoader.py         Dataset loading utilities  
EmbedNet.py              Embedding network wrapper  

loss/                    Loss implementations  
models/                  Backbone (ResNet18)  
optimizer/               SGD / Adam  
scheduler/               StepLR / CosineLR  

scripts/                 All training scripts (Train1 & Train2)
    exp01_train1.sh      Softmax baseline
    exp01_train2.sh      Softmax backbone + Config C
    exp02_train1.sh      AM-Softmax + SGD + StepLR
    exp02_train2.sh      Config A–D
    exp03_train1.sh      AM-Softmax + SGD + CosineLR
    exp03_train2.sh      Exp03 backbone + Config C
    exp04_train1.sh      Dropout backbone + class reduction
    exp04_train2.sh      Exp04-AM (Train2)
    exp04-AAM_train1.sh  Dropout backbone (same as Exp04)
    exp04-AAM_train2.sh  Final model (epoch 12)

exp04-AAM_test_ep12.csv  Final test output (submission)
```

## 3. How to Reproduce Experiments

```text
3.1 Train1 (from scratch)
Run the following depending on the experiment:

bash
bash scripts/exp01_train1.sh
bash scripts/exp02_train1.sh
bash scripts/exp03_train1.sh
bash scripts/exp04_train1.sh
3.2 Train2 (fine-tuning)
Run the corresponding Train2 script:

bash
bash scripts/exp01_train2.sh
bash scripts/exp02_train2.sh
bash scripts/exp03_train2.sh
bash scripts/exp04_train2.sh
bash scripts/exp04-AAM_train2.sh   # Final Model
The best final model is:

Exp04-AAM (Train2 epoch 12)
```


## 4. Running Test Evaluation
```text
Use the following command to generate the test CSV file:

bash
python trainEmbedNet.py \
  --gpu <GPU_ID> \
  --eval \
  --initial_model <PATH_TO_EXP04_AAM_EPOCH12> \
  --test_path /mnt/home/ee40034/data/test \
  --test_list /mnt/home/ee40034/data/test_pairs.csv \
  --output exp04-AAM_test_ep12.csv
```

## 5. Summary of Experimental Settings
```text
Exp01 – Softmax Baseline
Loss: Softmax

Optimizer: Adam

Scheduler: StepLR

Exp02 – Margin-Based Initialization
Loss: AM-Softmax

Optimizer: SGD

Scheduler: StepLR

Exp03 – Alternative Scheduler
Loss: AM-Softmax

Optimizer: SGD

Scheduler: CosineLR

Exp04 – Regularized Backbone + Class Reduction
Dropout: p = 0.4

Train1 classes: ~3000

Train2 classes: 949

Loss: AM-Softmax (Train1), AAM-Softmax (Train2)

Exp04-AAM – Final Model
Train1: AM-Softmax + SGD + StepLR

Train2: AAM-Softmax + SGD + CosineLR

Best epoch: 12

Test EER: 9.219%
```

## 6. Final Submitted Files
```text
exp04-AAM_test_ep12.csv

scripts/ (all training scripts)

All model, loss, scheduler, optimizer implementation files

No additional preprocessing scripts were used.
```

## 7. Reproducibility
All experimental results can be fully reproduced using the included scripts.
