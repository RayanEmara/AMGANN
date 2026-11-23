"""Four-channel matrix CNN, trained with torch and deployed with NumPy.

Artifacts contain weights and metadata only (npz, allow_pickle=False).
The NumPy forward pass allows PETSc containers to avoid a torch runtime.
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

VIEW_SIZE = 16
FEATURES = ["log2_ndof_over_16", "theta", "pressure_indicator", "log1p_horizon_over_8"]


def matrix_view(matrix, size=VIEW_SIZE):
    matrix = csr_matrix(matrix).copy()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    coo = matrix.tocoo()
    # Same balanced contiguous bins as common/view_maker.h, also for n < size.
    q, p = divmod(matrix.shape[0], size)
    cut = (q + 1) * p
    def bins(indices):
        if q == 0:
            return indices
        return np.where(indices < cut, indices // (q + 1), (indices - cut) // q + p)
    idx = bins(coo.row) * size + bins(coo.col)
    channels = np.zeros((4, size * size))
    np.add.at(channels[0], idx, coo.data)
    np.maximum.at(channels[1], idx, np.maximum(coo.data, 0))
    np.maximum.at(channels[2], idx, np.maximum(-coo.data, 0))
    np.add.at(channels[3], idx, 1)
    channels = np.sign(channels) * np.log1p(np.abs(channels))
    scale = np.max(np.abs(channels), axis=1, keepdims=True)
    channels /= np.where(scale > 0, scale, 1)
    return channels.reshape(4, size, size).astype(np.float32)


def parameters(n, theta, role, horizon):
    return np.array([np.log2(n) / 16, theta, float(role == "pressure"),
                     np.log1p(horizon) / 8], dtype=np.float32)


def build_model():
    import torch
    from torch import nn
    class CostCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
            self.fc1 = nn.Linear(8 * (VIEW_SIZE // 4)**2 + len(FEATURES), 32)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, image, param):
            x = nn.functional.avg_pool2d(torch.relu(self.conv1(image)), 2)
            x = nn.functional.avg_pool2d(torch.relu(self.conv2(x)), 2)
            x = torch.cat([x.flatten(1), param], dim=1)
            return self.fc2(torch.relu(self.fc1(x))).squeeze(1)
    return CostCNN()


def export_model(model, path, metadata):
    weights = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    metadata = {**metadata, "format_version": 1, "view_size": VIEW_SIZE, "features": FEATURES}
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **weights)


class Predictor:
    def __init__(self, path, backend):
        with np.load(path, allow_pickle=False) as artifact:
            self.metadata = json.loads(str(artifact["metadata"]))
            self.weights = {k: artifact[k].copy() for k in artifact.files if k != "metadata"}
        if (self.metadata["format_version"] != 1 or self.metadata["view_size"] != VIEW_SIZE
                or self.metadata["features"] != FEATURES):
            raise ValueError("Incompatible ANN artifact")
        if self.metadata["backend"] != backend:
            raise ValueError("Train and inference AMG backends must match; thresholds are not interchangeable")
        self.candidates = np.asarray(self.metadata["candidates"], dtype=np.float64)
        if not np.all((self.candidates > 0) & (self.candidates < 1)):
            raise ValueError("Invalid candidate thresholds")
        self.selections = []

    def forward(self, image, param):
        def conv(x, name):
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))), (3, 3), axis=(2, 3))
            return np.maximum(0, np.einsum("ncijhw,ochw->noij", windows,
                              self.weights[name + ".weight"], optimize=True)
                              + self.weights[name + ".bias"][None, :, None, None])
        def pool(x):
            n, c, h, w = x.shape
            return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
        x = pool(conv(image, "conv1"))
        x = pool(conv(x, "conv2"))
        x = np.concatenate([x.reshape(len(x), -1), param], axis=1)
        x = np.maximum(0, x @ self.weights["fc1.weight"].T + self.weights["fc1.bias"])
        return (x @ self.weights["fc2.weight"].T + self.weights["fc2.bias"]).ravel()

    def __call__(self, matrix, role, horizon):
        view = matrix_view(matrix)
        views = np.repeat(view[None], len(self.candidates), axis=0)
        params = np.stack([parameters(matrix.shape[0], t, role, horizon) for t in self.candidates])
        scores = self.forward(views, params)
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite ANN cost predictions")
        theta = float(self.candidates[np.argmin(scores)])
        self.selections.append({"role": role, "theta": theta, "scores": scores.tolist(),
                                "horizon": horizon})
        return theta


def train(dataset, output, epochs=150, seed=7):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    from copy import deepcopy
    dataset, output = Path(dataset), Path(output)
    if epochs < 1 or output.suffix != ".npz":
        raise ValueError("Require epochs >= 1 and an output filename ending in .npz")
    metadata = json.loads((dataset / "metadata.json").read_text())
    with np.load(dataset / "training.npz", allow_pickle=False) as data:
        images = torch.from_numpy(data["images"])
        params = torch.from_numpy(data["params"])
        targets = torch.from_numpy(data["targets"])
        train_mask = torch.from_numpy(data["split"] == "train")
        validation_mask = torch.from_numpy(data["split"] == "validation")
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Both training and validation matrix groups are required")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history, best, best_weights = [], float("inf"), None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.mean((model(images[train_mask], params[train_mask]) - targets[train_mask])**2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(images[validation_mask], params[validation_mask]) - targets[validation_mask])**2).item()
        history.append({"epoch": epoch + 1, "train_mse": loss.item(), "validation_mse": val})
        if val < best:
            best, best_weights = val, deepcopy(model.state_dict())
    if best_weights is None:
        raise RuntimeError("Training did not produce finite validation loss")
    model.load_state_dict(best_weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_model(model, output, {"backend": metadata["backend"],
                 "candidates": metadata["candidates"], "seed": seed,
                 "dataset": str(dataset.resolve()), "epochs": epochs,
                 "training_meshes": metadata["training_meshes"],
                 "validation_meshes": metadata["validation_meshes"]})
    deployed = Predictor(output, metadata["backend"])
    with torch.no_grad():
        reference = model(images, params).numpy()
    exported = deployed.forward(images.numpy(), params.numpy())
    np.testing.assert_allclose(exported, reference, atol=2e-6, rtol=2e-5)
    report = {"best_validation_mse": best, "samples": len(targets),
              "train_samples": int(train_mask.sum()), "validation_samples": int(validation_mask.sum()),
              "numpy_export_max_error": float(np.max(np.abs(reference - exported))),
              "history": history, "torch_version": torch.__version__}
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key != "history"}
