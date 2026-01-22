from __future__ import annotations

from pathlib import Path

import msgspec
import numpy as np


class ElectronStructureOutputs(msgspec.Struct, frozen=True):
    R: list[float]
    all_V: list[list[float]]
    all_identities: list[str]
    all_rho_alpha: list[list[float]]
    all_rho_beta: list[list[float]]

    @classmethod
    def from_json_file(cls, path: Path) -> ElectronStructureOutputs:
        with path.open("rb") as f:
            return msgspec.json.decode(f.read(), type=cls)

    def r_array(self) -> np.ndarray:
        return np.asarray(self.R)

    def v_array(self) -> np.ndarray:
        return np.asarray(self.all_V)

    def rho_alpha_array(self) -> np.ndarray:
        return np.asarray(self.all_rho_alpha)

    def rho_beta_array(self) -> np.ndarray:
        return np.asarray(self.all_rho_beta)
