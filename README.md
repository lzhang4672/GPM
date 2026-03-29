# Beyond Message Passing: Neural Graph Pattern Machine (GPM)

<div align='center'>

[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.6+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)
[![GPM arxiv](http://img.shields.io/badge/arxiv-2501.18739-yellow.svg)](https://arxiv.org/abs/2501.18739)
[![G2PM](https://img.shields.io/badge/G2PM-blue.svg)](https://github.com/zehong-wang/G2PM)

<img src="assets/logo.png" width='300'>
</div>

## 📝 Description

This is the official implementation of our ICML'25 paper [Beyond Message Passing: Neural Graph Pattern Machine](https://arxiv.org/abs/2501.18739). GPM represents a significant step towards next-generation graph learning backbone by moving beyond traditional message passing approaches.

### Key Features
- 🔍 Direct learning from graph substructures instead of message passing
- 🚀 Enhanced ability to capture long-range dependencies
- 💡 Efficient extraction and encoding of task-relevant graph patterns
- 🎯 Superior expressivity in handling complex graph structures

### Framework Overview
<img src="assets/workflow.png">
<img src="assets/framework.png">

GPM's workflow consists of three main steps:
1. Pattern extraction using random walk tokenizer
2. Pattern encoding via sequential modeling
3. Pattern processing through transformer encoder for downstream tasks

## 🛠️ Installation

### Prerequisites
- CUDA-compatible GPU (24GB memory minimum, 48GB recommended)
- CUDA 12.1
- Python 3.9+

### Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate GPM

# Install DGL
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# Install PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## 🚀 Quick Start
The code of GPM is presented in folder `/GPM`. You can run `main.py` and specify any dataset to run experiments. To ensure reproducability, we provide hyper-parameters in `config/main.yaml`. You can simply use command `--use_params` to set tuned hyper-parameters. 

### Basic Usage
```bash
# Run with default parameters
python GPM/main.py --dataset computers --use_params
```

### Supported Tasks & Datasets

1. **Node Classification**
   - `cora_full`, `computers`, `arxiv`, `products`
   - `wikics`, `deezer`, `blog`, `flickr`, `flickr_small`

2. **Link Prediction**
   - `link-cora`, `link-pubmed`, `link-collab`

3. **Graph Classification**
   - `imdb-b`, `collab`, `reddit-m5k`, `reddit-m12k`

4. **Graph Regression**
   - `zinc`, `zinc_full`

We also provide the interfaces of other widely used datasets like 
`photo`, `physics`, `reddit`, etc. Please check the datasets in `GPM/data/pyg_data_loader.py` for details. 



## 🔧 Configuration Options

### Training Parameters
- `--use_params`: Use tuned hyperparameters
- `--dataset`: Target dataset name
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--split`: Data split strategy (`public`, `low`, `median`, `high`)

### Model Architecture
- `--hidden_dim`: Hidden layer dimension
- `--heads`: Number of attention heads
- `--num_layers`: Number of Transformer layers
- `--dropout`: Dropout rate

### Pattern Configuration
- `--num_patterns`: Number of patterns per instance
- `--pattern_size`: Pattern size (random walk length)
- `--multiscale`: Range of walk lengths
- `--pattern_encoder`: Pattern encoder type (`transformer`, `mean`, `gru`)

For complete configuration options, please refer to our code documentation.

## 🔄 Domain Adaptation

Run domain adaptation experiments using:
```bash
python GPM/da.py --source acm --target dblp --use_params
```

Supported domain pairs:
- `acm -> dblp`, `dblp -> acm`
- `DE -> {EN, ES, FR, PT, RU}`

## 📊 GPM baseline coverage on 6 graph-classification datasets

### Slurm helper (single command)

If you want a single script that stages code and auto-recovers common torch/PyG binary mismatches,
use the provided job wrapper:

```bash
sbatch run_job.sh ba2motifs 5 1000
```

It will preflight torch with `scripts/diagnose_torch_segfault.py`; if `libze_loader.so.1` is missing it auto-adds the ComputeCanada ZE runtime path and retries.
If the env is still broken, it creates a fresh job-local venv and installs a pinned torch/PyG stack before launching training.

The installer ignores site-level pip config for the PyG extension step to avoid accidentally pulling incompatible `torch29` wheels from cluster wheelhouses.


To reproduce the reviewer-requested baseline protocol (8:2 train/test split, 5 seeds) for:
`Ba2Motifs`, `BAMultiShapes`, `Mutagenicity`, `BBBP`, `NCI1`, `PROTEINS`,
use:

```bash
python experiments/run_gpm_baseline.py --seeds 5 --gpu 0
```

This runner:
- uses `--split train80_test20`
- runs each dataset with `--split_repeat {3|5}`
- captures final `mean ± std` test metric and per-seed training time
- writes results to:
  - `artifacts/gpm_baseline_results.csv`
  - `artifacts/gpm_baseline_results.json`

If you prefer a quick check before full runs:

```bash
python experiments/run_gpm_baseline.py --seeds 3 --epochs 100
```

For cluster environments where compute jobs cannot access the internet, pre-cache datasets to a shared path first:

```bash
python scripts/precache_datasets.py \
  --data_path /shared/gpm_data \
  --pattern_path /shared/gpm_patterns
```

Then point training jobs to those persistent paths:

```bash
python GPM/main.py --dataset nci1 --split train80_test20 --split_repeat 5 \
  --data_path /shared/gpm_data --pattern_path /shared/gpm_patterns
```

If you hit dependency issues on a cluster, validate runtime dependencies first:

```bash
python scripts/check_runtime_deps.py
python scripts/diagnose_torch_segfault.py
```

If this reports a **CRASH** for `torch` or PyG modules, your environment likely has a binary mismatch
(e.g., CUDA/runtime ABI mismatch). In that case:

1. Recreate/clean the environment.
2. Install **torch** with the wheel/channel matching your cluster CUDA driver/runtime.
3. Install matching PyG wheels (`pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`).
4. Re-run `python scripts/check_runtime_deps.py`.


If `diagnose_torch_segfault.py` reports a crash/signal, do a clean reinstall of a **matching** torch/PyG stack
(example for CUDA 12.1):

```bash
python -m pip uninstall -y torch torchvision torchaudio torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv numpy
python -m pip install --no-cache-dir "numpy<2"
python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir torch-geometric==2.6.1
python -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python scripts/diagnose_torch_segfault.py
```

`requirements/runtime.txt` intentionally includes only lightweight packages and excludes torch/PyG binaries:

```bash
python -m pip install -r requirements/runtime.txt
```

## 📂 Repository Structure
```
└── GPM
    ├── GPM/              # Main package directory
    │   ├── data/         # Data loading and preprocessing
    │   ├── model/        # Model architectures
    │   ├── task/         # Task implementations
    │   ├── utils/        # Utility functions
    │   ├── main.py       # Main training script
    │   └── da.py         # Domain adaptation script
    ├── config/           # Configuration files
    ├── assets/           # Images and assets
    ├── data/             # Dataset storage
    ├── patterns/         # Extracted graph patterns
    └── environment.yml   # Conda environment spec
```

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025gpm,
  title={Beyond Message Passing: Neural Graph Pattern Machine},
  author={Wang, Zehong and Zhang, Zheyuan and Ma, Tianyi and Chawla, Nitesh V and Zhang, Chuxu and Ye, Yanfang},
  booktitle={Forty-Second International Conference on Machine Learning},
  year={2025}, 
}

@article{wang2025neural,
  title={Neural Graph Pattern Machine},
  author={Wang, Zehong and Zhang, Zheyuan and Ma, Tianyi and Chawla, Nitesh V and Zhang, Chuxu and Ye, Yanfang},
  journal={arXiv preprint arXiv:2501.18739},
  year={2025}
}
```

## 👥 Authors

- [Zehong Wang](https://zehong-wang.github.io/)
- [Zheyuan Zhang](https://jasonzhangzy1757.github.io/)
- [Tianyi Ma](https://tianyi-billy-ma.github.io/)
- [Nitesh V Chawla](https://niteshchawla.nd.edu/)
- [Chuxu Zhang](https://chuxuzhang.github.io/)
- [Yanfang Ye](http://yes-lab.org/)

For questions, please contact `zwang43@nd.edu` or open an issue.

## 🙏 Acknowledgements

This repository builds upon the excellent work from:
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://github.com/snap-stanford/ogb)
- [VQ](https://github.com/lucidrains/vector-quantize-pytorch)

We thank these projects for their valuable contributions to the field.
