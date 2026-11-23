import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ns_amg.network import Predictor, build_model, export_model, matrix_view, parameters
from ns_amg.pipeline import new_directory


def test_pooling_counts_are_not_value_channels():
    matrix = csr_matrix([[2.0, -1.0], [-1.0, 3.0]])
    view = matrix_view(matrix, size=4)
    assert np.isfinite(view).all()
    np.testing.assert_array_equal(view[3, :2, :2], np.ones((2, 2)))
    assert view[1, 0, 1] == 0 and view[2, 0, 1] == 1
    np.testing.assert_array_equal(matrix_view(csr_matrix((2, 2))), 0)


def test_numpy_inference_matches_trained_model_format(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = build_model().eval()
    path = tmp_path / "model.npz"
    export_model(model, path, {"backend": "pyamg", "candidates": [0.1, 0.5, 0.9]})
    predictor = Predictor(path, "pyamg")
    images = np.random.default_rng(0).normal(size=(3, 4, 16, 16)).astype(np.float32)
    params = np.stack([parameters(100, t, "pressure", 6) for t in (0.1, 0.5, 0.9)])
    with torch.no_grad():
        expected = model(torch.from_numpy(images), torch.from_numpy(params)).numpy()
    np.testing.assert_allclose(predictor.forward(images, params), expected, atol=1e-6)
    with pytest.raises(ValueError, match="backends must match"):
        Predictor(path, "petsc")


def test_existing_outputs_are_preserved(tmp_path):
    (tmp_path / "important.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        new_directory(tmp_path)
    assert (tmp_path / "important.txt").read_text() == "keep"
