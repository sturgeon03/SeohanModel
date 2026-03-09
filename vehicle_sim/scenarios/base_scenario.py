from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


class BaseScenario(ABC):
    """Minimal shared scenario interface."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self.config: dict[str, Any] = self.load_config(self.config_path)

    @staticmethod
    def load_config(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError

