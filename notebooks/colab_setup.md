# Google Colab Setup

This project can run in Google Colab, but the runtime must be configured from the Colab UI.

For the simplest workflow, open `notebooks/baseline_cnn_colab.ipynb` in Colab and run all cells. The notebook calls `run_baseline.py`, which reads settings from `experiments/baseline_cnn.yaml`.

## Runtime

1. Open a notebook in Google Colab.
2. Go to `Runtime > Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.
4. Run:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## Dependencies

From the project root in Colab, install the dependencies with `uv`:

```bash
pip install uv
uv sync
```

If you are running cells directly in a Colab notebook kernel, prefer:

```bash
pip install uv
uv pip install --system optuna numpy pandas matplotlib pyyaml tqdm
```

If you prefer plain `pip` in Colab, this also works:

```bash
pip install -r requirements.txt
```

## Dataset

The CIFAR-10 dataset is downloaded automatically the first time the dataloader is created:

```python
from data.dataloader import get_cifar10_dataloaders

train_loader, validation_loader, test_loader = get_cifar10_dataloaders(
    data_dir="data/raw",
    batch_size=64,
)
```
