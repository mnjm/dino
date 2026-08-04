# DINO - DIstillation with NO labels

A minimal implementation of [DINO](https://arxiv.org/pdf/2104.14294) self-supervised learning with a ViT-S/16 backbone. Trained from scratch and evaluated on the [Food-101](https://huggingface.co/datasets/ethz/food101) dataset.

## Evaluation

Weighted k-NN evaluation on Food-101 validation features:

| k | Top-1 accuracy | Top-5 accuracy |
| --- | --- | --- |
| 5 | 64.09% | 79.93% |
| 10 | 66.38% | 83.47% |
| 20 | 67.59% | 86.17% |
| 100 | 67.81% | 88.72% |

The exported ViT backbone is published at [here](https://huggingface.co/mnjm/DINOv1-ViT-S-16-food101).

![Attention heat map](https://raw.githubusercontent.com/mnjm/dino/refs/heads/assets/attn.png)

## Structure

- DINO model, projection head, loss, and ViT implementation in [`dino/`](./dino)
- Food-101 data pipeline and DINO multi-crop augmentation in [`data.py`](./data.py)
- Training entrypoint in [`train.py`](./train.py)
- Weighted k-NN evaluation in [`eval_knn.py`](./eval_knn.py)
- Hugging Face ViT exporter in [`hf_exporter.py`](./hf_exporter.py)
- Default and ViT-S/16 model configs in [`config/`](./config)

## Setup

Requirements:

- Python 3.12+
- `uv`

Install dependencies:

```bash
uv sync
```

## Dataset

The default dataset is `ethz/food101` from the Hugging Face Hub. It is downloaded automatically on the first training or evaluation run and cached under `./dataset/ethz/food101`.

DINO training uses the `train` split without its labels. k-NN evaluation indexes `train` features and reports accuracy on the `validation` split.

## Training

Run the default config:

```bash
uv run python train.py
```

Defaults are ViT-S/16, Food-101, CUDA, 300 epochs, `bf16`, `torch.compile`, and batch size 32. Outputs, configurations, and checkpoints are written under `logs/<run-name>/`.

Common Hydra overrides:

```bash
uv run python train.py device=auto dataloader.batch_size=16 n_epochs=100 torch_compile=false
```

See [`config/default.yaml`](./config/default.yaml) and [`config/model/vit-s-16.yaml`](./config/model/vit-s-16.yaml) for all settings. `device=auto` selects MPS, XLA, or CPU when CUDA is unavailable.

To log to Weights & Biases, add the following to `.env` and enable it:

```bash
WANDB_API_KEY=...
```

```bash
uv run python train.py wandb=true
```

## Evaluation

Evaluate the teacher backbone in a training checkpoint with similarity-weighted k-NN:

```bash
uv run python eval_knn.py ./logs/<run-name>/ViT-S-16.pth
```

The evaluator L2-normalizes CLS features, finds nearest training examples with FAISS inner-product search, and weights each class vote by `exp(similarity / temperature)`. It evaluates `k = 5, 10, 20, 100` by default and writes the index, labels, and validation embeddings to `./eval_embeddings`.

Useful overrides:

```bash
uv run python eval_knn.py ./logs/<run-name>/ViT-S-16.pth \
  --model-key student_model --nb-knn 10 20 100 --temperature 0.07
```

Reuse cached evaluation inputs:

```bash
uv run python eval_knn.py --load --nb-knn 5 10 20 100
```

## Hugging Face Export

Export the teacher ViT backbone from a training checkpoint in Transformers format:

```bash
uv run python hf_exporter.py \
  ./logs/<run-name>/ViT-S-16.pth \
  ./DINOv1-ViT-S-16-food101
```

Load the published model:

```python
from transformers import AutoImageProcessor, ViTModel

processor = AutoImageProcessor.from_pretrained("mnjm/DINOv1-ViT-S-16-food101")
model = ViTModel.from_pretrained("mnjm/DINOv1-ViT-S-16-food101")
inputs = processor(images=image, return_tensors="pt")
image_features = model(**inputs).last_hidden_state[:, 0]
```

## Citation

```text
@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```
