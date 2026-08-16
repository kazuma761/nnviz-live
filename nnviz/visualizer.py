"""
nnviz.visualizer
----------------
Main public API.  Users call ``Visualizer.run(model, dataloader)``
and get a live training window.
"""

import numpy as np
import torch
import torch.nn as nn

from .renderer import Renderer


def _softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _get_linear_layers(model):
    """Return all nn.Linear modules in forward order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _layer_sizes(model):
    """Infer (input_size, hidden..., output_size) from Linear layers."""
    layers = _get_linear_layers(model)
    if not layers:
        raise ValueError(
            "Model has no nn.Linear layers — nnviz requires at least "
            "one Linear layer."
        )
    sizes = [layers[0].in_features]
    for la in layers:
        sizes.append(la.out_features)
    return sizes


def _weight_matrices(model):
    """Return list of weight arrays (np.ndarray, shape out×in)."""
    return [la.weight.detach().cpu().numpy()
            for la in _get_linear_layers(model)]


class Visualizer:
    """
    Live neural-network training visualiser.

    Quickstart
    ----------
    >>> from nnviz import Visualizer
    >>> viz = Visualizer(input_shape=(28, 28),
    ...                  class_names=[str(i) for i in range(10)])
    >>> viz.run(model, train_loader, criterion, optimizer)

    Parameters
    ----------
    input_shape : tuple(int,int) | None
        (H, W) for image inputs; None for tabular data.
    class_names : list[str] | None
        Output class labels.
    max_visible_neurons : int
        Max neurons drawn per layer (large layers are sub-sampled visually).
    width, height : int
        Window size in pixels.
    fps : int
        Target render frame rate.
    batches_per_step : int
        Training batches processed per visual frame at speed ×1.
    normalise_input : bool
        Normalise the displayed pixel grid to [0,1] per sample.
        Does not affect training tensors.
    """

    def __init__(
        self,
        input_shape         = None,
        class_names         = None,
        max_visible_neurons = 24,
        width               = 1440,
        height              = 840,
        fps                 = 60,
        batches_per_step    = 1,
        normalise_input     = True,
    ):
        self.input_shape         = input_shape
        self.class_names         = class_names
        self.max_visible_neurons = max_visible_neurons
        self.width               = width
        self.height              = height
        self.fps                 = fps
        self.batches_per_step    = batches_per_step
        self.normalise_input     = normalise_input

        self._renderer    = None
        self._mode        = "stopped"
        self._speed       = 1
        self._batch_cnt   = 0
        self._epoch       = 0
        self._correct     = 0
        self._loss        = None
        self._last_pred   = None
        self._last_tgt    = None
        self._last_pixels = None
        self._last_probs  = None

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, model, dataloader, criterion, optimizer, epochs=999):
        """
        Open the visualiser window and train the model.

        Blocks until the window is closed or ``epochs`` is reached.

        Parameters
        ----------
        model       : nn.Module
        dataloader  : DataLoader   — yields (X_batch, y_batch)
        criterion   : loss fn      — e.g. nn.CrossEntropyLoss()
        optimizer   : optimiser    — e.g. optim.Adam(model.parameters())
        epochs      : int
        """
        sizes = _layer_sizes(model)

        self._renderer = Renderer(
            width       = self.width,
            height      = self.height,
            layer_sizes = sizes,
            input_shape = self.input_shape,
            class_names = self.class_names,
            max_visible = self.max_visible_neurons,
            fps         = self.fps,
        )

        running    = True
        epoch      = 0
        batch_iter = iter(dataloader)

        while running and epoch < epochs:

            # ── events ───────────────────────────────────────────────────
            for ev in self._renderer.poll_events():
                if ev == "quit":
                    running = False
                elif ev == "toggle":
                    self._mode = ("train"
                                  if self._mode == "stopped"
                                  else "stopped")
                elif ev == "speed_up":
                    self._speed = min(self._speed + 1, 8)
                elif ev == "speed_down":
                    self._speed = max(self._speed - 1, 1)

            if not running:
                break

            # ── training ─────────────────────────────────────────────────
            if self._mode == "train":
                n_steps = self.batches_per_step * self._speed
                for _ in range(n_steps):
                    try:
                        xb, yb = next(batch_iter)
                    except StopIteration:
                        epoch      += 1
                        self._epoch = epoch
                        if epoch >= epochs:
                            running = False
                            break
                        batch_iter = iter(dataloader)
                        xb, yb     = next(batch_iter)

                    loss, preds, probs = self._train_step(
                        model, xb, yb, criterion, optimizer
                    )
                    self._batch_cnt += 1
                    self._loss       = loss

                    pred_class      = int(preds[0])
                    true_class      = int(yb[0].item())
                    self._last_pred = pred_class
                    self._last_tgt  = true_class
                    if pred_class == true_class:
                        self._correct += 1
                    self._last_probs = probs[0]

                    # pixel display (first sample, flattened)
                    pix = (xb[0].detach().cpu().numpy()
                           .astype(np.float32).ravel())
                    if self.normalise_input:
                        mn, mx = pix.min(), pix.max()
                        if mx > mn:
                            pix = (pix - mn) / (mx - mn)
                    self._last_pixels = pix

                # push weights → triggers pulse animation
                self._renderer.update_weights(_weight_matrices(model))

            # ── render ───────────────────────────────────────────────────
            acc = (self._correct / self._batch_cnt * 100
                   if self._batch_cnt > 0 else None)

            state = dict(
                mode        = self._mode,
                batch       = self._batch_cnt,
                epoch       = self._epoch,
                loss        = self._loss,
                accuracy    = acc,
                correct     = self._correct,
                prediction  = self._last_pred,
                target      = self._last_tgt,
                class_names = self.class_names,
                speed       = self._speed,
            )

            self._renderer.draw(
                pixels      = self._last_pixels,
                predictions = self._last_probs,
                state       = state,
            )

        self._renderer.quit()

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _train_step(model, xb, yb, criterion, optimizer):
        optimizer.zero_grad()
        output = model(xb)
        loss   = criterion(output, yb)
        loss.backward()
        optimizer.step()

        logits = output.detach().cpu().numpy()
        loss_v = float(loss.detach().item())
        probs  = np.array([_softmax(logits[i])
                           for i in range(len(logits))])
        preds  = np.argmax(probs, axis=1)
        return loss_v, preds, probs