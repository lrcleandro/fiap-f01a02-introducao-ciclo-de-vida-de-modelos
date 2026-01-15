"""
Treina um modelo de Random Forest para o dataset de doenças cardíacas
e loga os resultados no MLflow.

"""

import sys
import os
import json
import warnings
import argparse
from datetime import datetime
import pandas as pd
import mlflow
from mlflow import sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Importar transformers customizados
from preprocessing import CategoricalEncoder, FeatureEngineer, MissingValueImputer
from mlflow_utils import resolve_tracking_paths

warnings.filterwarnings('ignore')

# ============================================================================
# Configurações de hiperparâmetros
# ============================================================================

# PARAMS = {
#     'max_depth': 5,
#     'max_features': 7,
#     'min_samples_split': 10,
#     'n_estimators': 50,
#     'random_state': 42
# }

# AJUSTE DE HIPERPARÂMETROS (descomente para usar) -------------------
PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42
}


# ============================================================================
# Escolha de hiperparâmetros ativos
# ============================================================================

# VERSÃO BASELINE (padrão para começar) ---------------------------
ACTIVE_PARAMS = PARAMS

# VERSÃO OTIMIZADA (descomente este bloco para usar) ---------------

def load_and_prepare_data(data_path):
    """Carrega e prepara os dados para treinamento."""
    print(f"Carregando dados de: {data_path}")
    df = pd.read_csv(data_path)
    
    # Criar target binário
    if 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
    
    # Remover colunas de metadados
    columns_to_drop = ['id', 'dataset', 'num']
    cols_dropped = [col for col in columns_to_drop if col in df.columns]
    if cols_dropped:
        df = df.drop(columns=cols_dropped)
        print(f"Colunas removidas: {cols_dropped}")
    
    # Separar features e target
    X = df.drop(columns=['target'], errors='ignore')
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def create_pipeline(params, numeric_cols, categorical_cols):
    """Cria pipeline completo com pré-processamento e modelo."""
    pipeline = Pipeline([
        ('imputer', MissingValueImputer(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )),
        ('categorical_encoding', CategoricalEncoder()),
        ('feature_engineering', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**params))
    ])
    return pipeline


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Avalia o modelo e retorna métricas."""
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    return metrics


def train_model(params=None, data_path='../data/heart_disease_uci.csv',
                mlflow_experiment='heart-disease-cicd'):
    """
    Treina modelo e loga no MLflow.
    
    Args:
        params: dicionário de hiperparâmetros. Se None usa ACTIVE_PARAMS.
        data_path: caminho para o dataset.
        mlflow_experiment: nome do experimento MLflow.
    """
    # Selecionar hiperparâmetros ativos e detectar variante cedo
    if params is None:
        params = ACTIVE_PARAMS
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_name = "random_forest_training"

    print(f"\n{'='*60}")
    print("Iniciando treinamento de Random Forest")
    print(f"{'='*60}\n")

    # Carregar dados
    X, y = load_and_prepare_data(data_path)

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # Detectar colunas categóricas e numéricas
    all_cols = X_train.columns.tolist()
    categorical_cols = [c for c in ['sex', 'cp', 'restecg', 'slope', 'thal'] if c in all_cols]
    numeric_cols = [c for c in all_cols if c not in categorical_cols]    
   
    # Criar e treinar pipeline
    pipeline = create_pipeline(params, numeric_cols, categorical_cols)
    print("Treinando pipeline completo...")
    pipeline.fit(X_train, y_train)
    
    # Avaliar modelo
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    print(f"\nMétricas de avaliação:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    tracking_uri, tracking_dir = resolve_tracking_paths()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Logar no MLflow
    mlflow.set_experiment(mlflow_experiment)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Logar parâmetros
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param('model_family', 'RandomForest')
        
        # Logar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Criar signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        # Logar modelo
        model_info = mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            code_paths=["preprocessing.py"],
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        print(f"\n✓ Modelo logado no MLflow")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model URI: {model_info.model_uri}")

        metadata = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "test_accuracy": metrics['test_accuracy'],
            "logged_at": datetime.utcnow().isoformat()
        }

        metadata_path = os.environ.get("MLFLOW_LATEST_RUN_FILE")
        if not metadata_path and tracking_dir:
            metadata_path = os.path.join(tracking_dir, "latest_run.json")

        if metadata_path:
            if not os.path.isabs(metadata_path):
                metadata_path = os.path.join(script_dir, metadata_path)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as fp:
                json.dump(metadata, fp, indent=2)
            print(f"  Metadata salva em: {metadata_path}")
            print("  ➜ Execute register_model.py para atualizar o Model Registry.")
        else:
            print("  ⚠ Não foi possível determinar caminho para salvar metadata do run.")
    
    print(f"\n{'='*60}")
    print(f"Treinamento concluído!")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar modelo Heart Disease')
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/heart_disease_uci.csv',
        help='Caminho para o dataset'
    )

    args = parser.parse_args()

    metrics = train_model(
        data_path=args.data_path
    )

    # Opcional: falhar em caso de acurácia extremamente baixa
    if metrics['test_accuracy'] < 0.50:
        print("⚠ ERRO: Acurácia muito baixa! Verifique dados ou hiperparâmetros.")
        sys.exit(1)
