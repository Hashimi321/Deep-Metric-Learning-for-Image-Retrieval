# Deep Metric Learning for Image Retrieval
## SaadAli_MSDS25066_03 | Deep Learning Spring 2026

---

## Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```
SaadAli_MSDS25066_03/
├── model.py              # EmbeddingNet: ResNet-50 + projection head + L2 norm
├── dataset.py            # ContrastiveDataset and TripletDataset
├── loss.py               # ContrastiveLoss, TripletLoss, batch_hard_mining
├── train.py              # Training loop for all 3 experiments
├── inference.py          # Load model and generate embeddings for new images
├── save_embeddings.py    # Precompute and save embeddings to disk
├── retrieval.py          # Recall@K evaluation, t-SNE, retrieval visualization
├── split_dataset.py      # Split raw Caltech-101 into train/val/test
├── embeddings/           # Saved .npy embedding files
├── weights/              # Saved model checkpoints (.pt files)
│   ├── exp1/             # Contrastive loss checkpoints
│   ├── exp2/             # Triplet random checkpoints
│   └── exp3/             # Triplet hard mining checkpoints
├── graphs/               # t-SNE plots and retrieval grid images
├── Report.pdf            # Assignment report
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Step 1 — Split Dataset
```bash
python split_dataset.py --src "caltech-101" --dst "caltech_split"
```

This creates `caltech_split/train/`, `caltech_split/val/`, `caltech_split/test/`
with a stratified 70/15/15 split across all 102 classes.

---

## Step 2 — Train

Run all three experiments:
```bash
python train.py --data_path "caltech_split" --experiment all
```

Run a single experiment:
```bash
python train.py --data_path "caltech_split" --experiment exp1
python train.py --data_path "caltech_split" --experiment exp2
python train.py --data_path "caltech_split" --experiment exp3
```

Experiments:
- `exp1` → Contrastive Loss, Random Pairs
- `exp2` → Triplet Loss, Random Triplets
- `exp3` → Triplet Loss, Hard Negative Mining

Checkpoints saved to `weights/expN/` after every epoch.
Training log saved to `weights/expN/training_log.csv`.

---

## Step 3 — Save Embeddings

Run after training to precompute embeddings for all splits:
```bash
python save_embeddings.py \
    --model_path "weights/exp1/exp1_contrastive_epoch10.pt" \
    --data_path  "caltech_split" \
    --exp_name   "exp1_contrastive"

python save_embeddings.py \
    --model_path "weights/exp2/exp2_triplet_random_epoch10.pt" \
    --data_path  "caltech_split" \
    --exp_name   "exp2_triplet_random"

python save_embeddings.py \
    --model_path "weights/exp3/exp3_triplet_hard_epoch10.pt" \
    --data_path  "caltech_split" \
    --exp_name   "exp3_triplet_hard"
```

Embeddings saved to `embeddings/` as `.npy` files.

---

## Step 4 — Evaluate and Visualize
```bash
python retrieval.py \
    --embeddings_path "embeddings/exp1_contrastive_test_embeddings.npy" \
    --labels_path     "embeddings/exp1_contrastive_test_labels.npy" \
    --data_path       "caltech_split/test" \
    --exp_name        "exp1_contrastive" \
    --model_path      "weights/exp1/exp1_contrastive_epoch10.pt"
```

Generates:
- Recall@1 and Recall@5 scores
- t-SNE visualization saved to `graphs/`
- 10 retrieval grid images saved to `graphs/`

---

## Step 5 — Inference on New Images

Single image:
```bash
python inference.py \
    --model_path "weights/exp1/exp1_contrastive_epoch10.pt" \
    --image_path "caltech-101/accordion/image_0001.jpg"
```

Multiple images:
```bash
python inference.py \
    --model_path  "weights/exp1/exp1_contrastive_epoch10.pt" \
    --image_path  "img1.jpg" "img2.jpg" "img3.jpg" \
    --save_output "my_embeddings.npy"
```

---

## Results

| Experiment | Loss Function | Sampling | Recall@1 | Recall@5 |
|---|---|---|---|---|
| Exp1 | Contrastive | Random Pairs | 85.27% | 94.07% |
| Exp2 | Triplet | Random Triplets | 59.41% | 77.90% |
| Exp3 | Triplet | Hard Neg Mining | 80.42% | 90.79% |

---

## Notes

- Training was performed on Google Colab with Tesla T4 GPU
- All three experiments use identical hyperparameters (lr=1e-4, epochs=10, batch=64)
- Model weights are saved to Google Drive — see shareable link in Report.pdf
- GitHub repository shared with TA: HajraZawwar
