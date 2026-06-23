# GeoCAN: Nonlinear Causal-Geometric Learning for Echocardiography Quality Assessment

This repository provides a public implementation of **GeoCAN** for echocardiography quality assessment.

<img width="2368" height="1072" alt="Fig2-1_GeoCAN" src="https://github.com/user-attachments/assets/5c8d84b7-1120-4841-bcc0-e64952a70778" />

GeoCAN contains two main components:

1. **Causal-Geometric Learning (CGL)**  
   It estimates a directed causal interaction map with a generalized Lehmer mean, derives asymmetric causal effects from `C - C^T`, and modulates visual factor maps.

2. **Nonlinear Relational Aggregation (NRA)**  
   It encodes causal-enhanced visual tokens with a Transformer and performs nonlinear token-level residual aggregation using learnable piecewise-linear spline functions.

<img width="2381" height="1069" alt="Fig1-1_Overview" src="https://github.com/user-attachments/assets/c2ea2f32-7972-4f4b-988a-95d6af081d7b" />


## Important note about the graph-theoretic ranking loss

The spectral graph/BPR-style geometric ranking component used for `L_geo` is **not included in this public release** because its implementation is derived from third-party code that is not permitted to be publicly redistributed.

The omitted component is based on:

```bibtex
@inproceedings{
zheng2026graphtheoretic,
title={Graph-Theoretic Insights into Bayesian Personalized Ranking for Recommendation},
author={Kai Zheng and Jianxin Wang and Jinhui Xu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=tmtUA2X57D}
}
```

Please contact the original authors for access to the official graph-theoretic BPR / spectral ranking implementation.  
In this public repository, the code keeps the interface and the hyperbolic node construction pipeline, but the restricted ranking loss itself is replaced with a placeholder.

## Repository structure

```text
GeoCAN_public_release/
├── geocan/
│   ├── __init__.py
│   ├── model.py                  # GeoCAN model: CGL + NRA + hyperbolic interface
│   ├── pooling.py                # Generalized Lehmer causality modules
│   ├── restricted_graph_bpr.py    # placeholder for restricted L_geo implementation
│   └── data.py                   # dataset and split utilities
├── scripts/
│   ├── train.py                  # public training script
│   └── test.py                   # public testing script
├── examples/
│   └── minimal_forward.py         # forward-pass sanity check
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Data format

The public scripts expect the following CACTUS-style folder organization:

```text
Images Dataset/
├── A4C/
├── PL/
├── PSAV/
├── PSMV/
├── Random/
└── SC/

Grades/
├── xxx_grades.csv
└── ...
```

Each CSV should contain columns equivalent to:

```text
Image Name, Subfolder, Grade
```

Accepted aliases include `ImageName`, `image_name`, `Subfolder Name`, `SubFolder`, and `grade`.

Grades are expected to be integer labels from `0` to `10`.

## Training

By default, the public training script trains GeoCAN without the restricted `L_geo` term:

```bash
python scripts/train.py \
  --image-dir "Images Dataset" \
  --grade-dir "Grades" \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-path model.pt
```

To run the exact full method with the graph-theoretic ranking loss, obtain the authorized implementation from the original authors and replace the placeholder in `geocan/restricted_graph_bpr.py`.

## Testing

```bash
python scripts/test.py \
  --image-dir "Images Dataset" \
  --grade-dir "Grades" \
  --checkpoint model.pt \
  --output-csv predictions.csv
```

## Notes

- Images are resized to `224 x 224` in the public scripts. If you already crop the fan-shaped acoustic region before training, keep this preprocessing consistent with the paper.
- The model supports view classification and quality classification.
- The public code preserves the key CGL and NRA structure while omitting only the restricted graph-theoretic BPR spectral ranking implementation.
