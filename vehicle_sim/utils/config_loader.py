"""
Configuration Loader Utility
YAML 설정 파일을 로드하는 공통 유틸리티
"""

import yaml
from pathlib import Path
from typing import Dict, Optional


def load_param(module_name: str, config_path: Optional[str] = None) -> Dict:
    """
    차량 모듈별 파라미터 로드

    Args:
        module_name: 모듈 이름 ('brake', 'motor', 'suspension', 'steering', 'tire', 'vehicle_body')
        config_path: YAML 파일 경로. None이면 기본 vehicle_standard.yaml 사용

    Returns:
        Dict: 해당 모듈의 파라미터 딕셔너리

    Example:
        >>> brake_param = load_param('brake')
        >>> motor_param = load_param('motor')
        >>> custom_param = load_param('brake', '/path/to/custom.yaml')
    """
    if config_path is None:
        # 기본 경로: vehicle_sim/models/params/vehicle_standard.yaml
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        config_path = project_root / "models" / "params" / "vehicle_standard.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    if not full_config:
        return {}

    return full_config.get(module_name, {})
