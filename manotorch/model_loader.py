"""Trusted MANO pickle loading without the legacy Chumpy object graph.

The licensed MANO model is supplied by the user under ``assets/mano``.  This
loader therefore accepts only a trusted local file; never point it at an
untrusted pickle received from a user or network service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ManoModelParameters:
    """Raw arrays required by :class:`manotorch.manolayer.ManoLayer`."""

    shapedirs: np.ndarray
    posedirs: np.ndarray
    v_template: np.ndarray
    joint_regressor: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
    kintree_table: np.ndarray
    hands_components: np.ndarray
    hands_mean: np.ndarray


_REQUIRED_FIELDS = frozenset(
    {
        "shapedirs",
        "posedirs",
        "v_template",
        "J_regressor",
        "weights",
        "f",
        "kintree_table",
        "hands_components",
        "hands_mean",
    }
)


def _as_dense_array(value: Any) -> np.ndarray:
    """Convert a NumPy or SciPy sparse model field to a dense array."""
    to_array = getattr(value, "toarray", None)
    return np.array(to_array() if callable(to_array) else value, copy=True)


def load_mano_model(model_path: str | Path) -> ManoModelParameters:
    """Load the subset of a trusted MANO pickle used by the Torch LBS layer.

    Chumpy was formerly used here to materialize a differentiable SMPL graph.
    ``ManoLayer`` only registers the static model arrays and performs all LBS
    operations in Torch, so materializing that graph is unnecessary.
    """
    path = Path(model_path)
    with path.open("rb") as handle:
        raw_model = pickle.load(handle, encoding="latin1")
    if not isinstance(raw_model, Mapping):
        raise TypeError(f"expected a mapping in MANO model {path}, got {type(raw_model).__name__}")

    missing = sorted(_REQUIRED_FIELDS.difference(raw_model))
    if missing:
        raise KeyError(f"MANO model {path} is missing required fields: {', '.join(missing)}")

    return ManoModelParameters(
        shapedirs=np.array(raw_model["shapedirs"], copy=True),
        posedirs=np.array(raw_model["posedirs"], copy=True),
        v_template=np.array(raw_model["v_template"], copy=True),
        joint_regressor=_as_dense_array(raw_model["J_regressor"]),
        weights=np.array(raw_model["weights"], copy=True),
        faces=np.array(raw_model["f"], copy=True),
        kintree_table=np.array(raw_model["kintree_table"], copy=True),
        hands_components=np.array(raw_model["hands_components"], copy=True),
        hands_mean=np.array(raw_model["hands_mean"], copy=True),
    )
