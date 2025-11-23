"""Etapa de registro/promoção no Model Registry usando metadata do último run."""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_utils import resolve_tracking_paths


def load_metadata(metadata_path: Optional[str], tracking_dir: Optional[str]) -> str:
    """Retorna o caminho para o arquivo com info do último run."""
    if metadata_path:
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), metadata_path)
    elif tracking_dir:
        metadata_path = os.path.join(tracking_dir, "latest_run.json")
    else:
        raise FileNotFoundError(
            "Não foi possível determinar o caminho da metadata. "
            "Defina --metadata-path ou MLFLOW_TRACKING_FOLDER."
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Arquivo de metadata não encontrado: {metadata_path}")
    return metadata_path


def wait_for_version_files(tracking_dir: Optional[str], model_name: str, version: str, timeout: float = 10.0) -> None:
    """Garante que o meta.yaml da versão exista antes de continuar."""
    if not tracking_dir:
        return
    models_root = os.path.join(tracking_dir, "models")
    if not os.path.exists(models_root):
        # Tracking remoto (sem diretórios locais). Não aguardar.
        return
    meta_path = os.path.join(
        tracking_dir,
        "models",
        model_name,
        f"version-{version}",
        "meta.yaml",
    )
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if os.path.exists(meta_path):
            return
        time.sleep(0.5)
    print(
        f"  ⚠ Aviso: arquivo '{meta_path}' não apareceu após {timeout}s. "
        "Prosseguindo mesmo assim."
    )


def cleanup_orphan_versions(tracking_dir: Optional[str], model_name: str) -> None:
    if not tracking_dir:
        return
    models_root = os.path.join(tracking_dir, "models", model_name)
    if not os.path.exists(models_root):
        return
    for entry in os.listdir(models_root):
        if not entry.startswith("version-"):
            continue
        version_dir = os.path.join(models_root, entry)
        meta_file = os.path.join(version_dir, "meta.yaml")
        if os.path.isdir(version_dir) and not os.path.exists(meta_file):
            print(f"  ⚠ Removendo diretório órfão: {version_dir}")
            shutil.rmtree(version_dir, ignore_errors=True)


def register_and_promote(model_name: str, metadata_path: Optional[str]) -> None:
    tracking_uri, tracking_dir = resolve_tracking_paths()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    metadata_file = load_metadata(metadata_path, tracking_dir)
    print(f"Carregando metadata: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    run_id = data.get('run_id')
    model_uri = data.get('model_uri')
    test_accuracy = data.get('test_accuracy')

    if not run_id or not model_uri:
        raise ValueError("Metadata incompleta: run_id ou model_uri ausentes.")

    cleanup_orphan_versions(tracking_dir, model_name)
    print("Registrando nova versão no Model Registry...")
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_for_version_files(tracking_dir, model_name, model_version.version)
    print(f"  Modelo: {model_name} | Versão criada: {model_version.version}")

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    prev_versions = [v for v in versions if v.version != model_version.version]

    def set_alias(version: str) -> None:
        try:
            client.set_registered_model_alias(model_name, "Production", version)
            print(f"  ✓ Alias 'Production' atualizado para versão {version}.")
        except Exception as exc:
            print(f"  ⚠ Falha ao atualizar alias Production: {exc}")

    if not prev_versions:
        print("  Primeira versão registrada. Promovendo para Production.")
        set_alias(model_version.version)
        return

    # Encontrar melhor acurácia anterior
    best_prev_acc = None
    best_prev_version = None
    for version in prev_versions:
        run_id_prev = getattr(version, 'run_id', None)
        if not run_id_prev:
            continue
        try:
            run_prev = client.get_run(run_id_prev)
            acc_prev = run_prev.data.metrics.get('test_accuracy')
            if acc_prev is not None and (best_prev_acc is None or acc_prev > best_prev_acc):
                best_prev_acc = acc_prev
                best_prev_version = version.version
        except Exception:
            continue

    if best_prev_acc is None or test_accuracy is None:
        print("  Não foi possível comparar acurácias. Mantendo Production atual.")
        return

    print(f"  Melhor acurácia anterior: {best_prev_acc:.4f} (versão {best_prev_version})")
    if test_accuracy > best_prev_acc:
        set_alias(model_version.version)
        print(
            f"  ✓ Versão {model_version.version} promovida a Production "
            f"(acurácia {test_accuracy:.4f} > {best_prev_acc:.4f})."
        )
    else:
        print(
            f"  ✗ Acurácia {test_accuracy:.4f} não supera {best_prev_acc:.4f}. "
            "Mantém Production existente."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Registrar e promover modelo no MLflow Model Registry')
    parser.add_argument(
        '--metadata-path',
        type=str,
        help='Caminho explícito para o arquivo latest_run.json (opcional)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='heart-disease-model',
        help='Nome do modelo no Model Registry'
    )
    args = parser.parse_args()

    try:
        register_and_promote(model_name=args.model_name, metadata_path=args.metadata_path)
    except Exception as exc:
        print(f"⚠ ERRO durante registro/promocao: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
