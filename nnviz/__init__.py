"""
nnviz
=====
Live neural-network training visualiser for PyTorch.

Quickstart
----------
>>> import torch.nn as nn, torch.optim as optim
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from nnviz import Visualizer
>>>
>>> # 1. Define any model with nn.Linear layers
>>> model = nn.Sequential(nn.Linear(784,128), nn.ReLU(),
...                        nn.Linear(128,64),  nn.ReLU(),
...                        nn.Linear(64,10))
>>>
>>> # 2. Wrap your data in a DataLoader
>>> loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64)
>>>
>>> # 3. Run — window opens, press T to start training
>>> viz = Visualizer(input_shape=(28,28),
...                  class_names=[str(i) for i in range(10)])
>>> viz.run(model, loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters()))
"""

from .visualizer import Visualizer   # noqa: F401
from .renderer   import Renderer     # noqa: F401 (advanced use)

__version__ = "0.1.0"
__all__     = ["Visualizer", "Renderer"]
