"""Utilitários compartilhados para configuração do MLflow."""

import os
from typing import Optional, Tuple


def resolve_tracking_paths() -> Tuple[str, Optional[str]]:
    """Retorna (tracking_uri, tracking_dir) usando env vars ou defaults."""
    tracking_uri_env = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_dir: Optional[str] = None

    if tracking_uri_env:
        tracking_uri = tracking_uri_env
        if tracking_uri.startswith("file://"):
            tracking_dir = tracking_uri[len("file://") :]
    else:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        tracking_folder = os.environ.get("MLFLOW_TRACKING_FOLDER")
        if not tracking_folder:
            tracking_folder = "mlruns_ci" if os.environ.get("CI") else "mlruns"
        if not os.path.isabs(tracking_folder):
            tracking_folder = os.path.join(repo_dir, tracking_folder)
        os.makedirs(tracking_folder, exist_ok=True)
        tracking_dir = tracking_folder
        tracking_uri = f"file://{tracking_dir}"

    return tracking_uri, tracking_dir
