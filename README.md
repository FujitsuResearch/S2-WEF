# S2-WEF: Dynamic Free-Rider Detection in Cross-Silo Federated Learning

## Paper

- https://arxiv.org/abs/2604.04611

## Repository layout

```
.
├── run.py                     Experiment driver
├── s2wef/                     Our code (BSD 3-Clause Clear)
│   ├── pipeline.py              S2-WEF detection, one global round
│   ├── wef.py                   WEF-matrix construction
│   ├── clustering.py            Robust standardization, Ward clustering
│   ├── detector.py              Vote-based decision
│   ├── attacks.py               RWA, SPA, DWA, ADWA, AWCA
│   ├── participant.py           Local training and update submission
│   ├── aggregator.py            Global model averaging and evaluation
│   ├── partition.py             Non-IID partitioning policy
│   ├── datasets.py              Dataset loading, per-class indices
│   ├── adult.py                 Adult census preprocessing
│   ├── models.py                LeNet, MLP, model selection
│   ├── metrics.py               Result logging
│   ├── visualization.py         Optional figures
│   └── conf.json                Experiment configuration
├── third_party/               Code from other projects (MIT)
│   ├── niid_bench/              Dirichlet label-skew sampling
│   └── pytorch_cifar/           ResNet for 32x32 inputs
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```
## License

This repository is published under more than one license.

| Path | License | Copyright |
| --- | --- | --- |
| `run.py`, `s2wef/**` | BSD 3-Clause Clear | Fujitsu Limited |
| `third_party/niid_bench/**` | MIT | Yiqun Diao, Qinbin Li |
| `third_party/pytorch_cifar/**` | MIT | see the file |

Every source file carries an `SPDX-License-Identifier` tag. The full text of
the BSD 3-Clause Clear License is in `LICENSE`; the license of each
third-party component is in the `LICENSE` file of its directory and is also
reproduced in `THIRD_PARTY_NOTICES.md`. Nothing in this repository alters
the terms under which the third-party components were received.

## Requirements

Python 3.10 or later, with:

```
torch
torchvision
numpy
scipy
scikit-learn
pandas
matplotlib
```

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib
```

## Datasets

MNIST, CIFAR-10 and CIFAR-100 are downloaded to `./data/` on first use.

Adult is not downloaded automatically by torchvision. Place
`adult.data` and `adult.test` from the
[UCI repository](https://archive.ics.uci.edu/dataset/2/adult) under
`./data/adult_datasets/`; if they are absent, `s2wef/adult.py` falls back to
fetching them from the UCI URLs. 

## Running an experiment

```bash
python run.py                       # uses ./s2wef/conf.json
python run.py -c path/to/conf.json  # or a configuration of your choice
```

Per-round progress is printed to stdout; one row per run is appended to
`result/YYYY_MMDD.csv`, holding every configuration key alongside the
detection accuracy, precision, recall, F1, false-positive rate, final global
accuracy and wall-clock time. 

### Configuration

Dataset and model must be set together, because the WEF-matrix is read from
a parameter tensor selected by index:

| `dataset` | `model_name` | `WEF-layer` | `lr` | `momentum` | `global_epochs` |
| --- | --- | --- | --- | --- | --- |
| `mnist` | `LeNet` | `-4` | 0.005 | 0.0001 | 50 |
| `adult` | `mlp` | `-4` | 0.0001 | 0.0001 | 50 |
| `cifar10` | `resnet` | `-2` | 0.01 | 0.9 | 80 |
| `cifar100` | `resnet100` | `-2` | 0.01 | 0.9 | 100 |

Attack and free-rider behaviour:

| Key | Values | Meaning |
| --- | --- | --- |
| `free_riders_type` | `RWA`, `SPA`, `DWA`, `ADWA`, `AWCA` | Which attack the free-riders run |
| `free_riders_state` | `dynamic` | Fixed attackers, free-riding from `free_riders_start_round` to the end |
| | `dynamic_random` | A fresh random subset attacks in each round from `attack_start_round` |
| | `static` | Fixed attackers, free-riding in every round |
| | `NoFreeRider` | No attack; used to measure the false-positive rate |
| `k`, `free_riders` | integers | Benign and free-riding client counts; their sum is the federation size |
| `sigma` | float | `R` for RWA, noise standard deviation for ADWA and AWCA |
| `free_riders_start_round`, `attack_start_round` | integer, 1-based | First round in which an attack may occur. Keep at 3 or above: DWA, ADWA and AWCA need two previously received global models |

Detection:

| Key | Values | Meaning |
| --- | --- | --- |
| `method` | `proposed`, `no_defense` | Whether detected clients are excluded from aggregation |
| `decision_source` | `pipeline` | Full S2-WEF: clustering plus the vote |
| | `gamma_only`, `dev_only`, `cluster_only` | Single components, for the ablation study |
| `gamma_threshold_k` | float | `k` in the similarity threshold. 5 in the paper |
| `ablation` | `cos/l2`, `cos/l1`, `cos` | How to Calculate the similarity score $\gamma$ |
| `majority_vote` | `yes`, `no` | Whether the suspicious cluster is verified before labelling |
| `dev_threshold_eps` | float | Epsilon of the deviation threshold at `dev_threshold_ref_n` clients; scaled by client count |
| `no_iid` | `yes`, `no` | Dirichlet partition, or class-balanced IID sampling |
| `beta` | float | Dirichlet concentration. 0.5 in the paper |
| `random_seed` | integer | Seeds NumPy, PyTorch and the standard library |
| `verbose` | `yes`, `no` | Whether the per-client score lists are printed |

## Citation

```bibtex
@misc{nakamura2026s2wef,
  title         = {Dynamic Free-Rider Detection in Cross-Silo Federated
                   Learning via Simulated Attack Patterns},
  author        = {Nakamura, Motoki},
  year          = {2026},
  eprint        = {2604.04611},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2604.04611}
}
```
