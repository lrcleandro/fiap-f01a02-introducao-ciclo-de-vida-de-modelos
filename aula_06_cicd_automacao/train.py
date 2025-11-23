"""
Treina um modelo de Random Forest para o dataset de doenças cardíacas
e loga os resultados no MLflow.

"""

import sys
import os
import warnings
import argparse
import pandas as pd
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Importar transformers customizados
from preprocessing import CategoricalEncoder, FeatureEngineer, MissingValueImputer

warnings.filterwarnings('ignore')

# ============================================================================
# Configurações de hiperparâmetros
# ============================================================================


PARAMS = {
    'max_depth': 5,
    'max_features': 7,
    'min_samples_split': 10,
    'n_estimators': 50,
    'random_state': 42
}

# AJUSTE DE HIPERPARÂMETROS (descomente para usar) -------------------
# PARAMS = {
#     'n_estimators': 100,
#     'max_depth': None,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'max_features': 'sqrt',
#     'random_state': 42
# }


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
    
    # Configurar tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        tracking_folder = os.environ.get("MLFLOW_TRACKING_FOLDER")
        if not tracking_folder:
            tracking_folder = "mlruns_ci" if os.environ.get("CI") else "mlruns"
        tracking_path = tracking_folder
        if not os.path.isabs(tracking_path):
            tracking_path = os.path.join(repo_dir, tracking_path)
        os.makedirs(tracking_path, exist_ok=True)
        tracking_uri = f"file://{tracking_path}"
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
        
        # Registro sempre realizado (baseline ou nova versão)
        test_accuracy = metrics['test_accuracy']
        print("\nRegistrando versão no Model Registry...")
        try:
            model_name = "heart-disease-model"
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=model_name
            )
            print(f"  Modelo registrado: {model_name} | Versão: {model_version.version}")

            # Comparar com versões anteriores para possível promoção
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            prev_versions = [v for v in versions if v.version != model_version.version]

            if not prev_versions:
                print("  Primeira versão registrada. Promovendo para Production.")
                try:
                    client.set_registered_model_alias(
                        model_name,
                        "Production",
                        model_version.version
                    )
                    print(f"  ✓ Versão {model_version.version} promovida como alias 'Production'.")
                except Exception as e:
                    print(f"  ⚠ Falha ao definir alias Production: {e}")
            else:
                # Obter melhor acurácia anterior
                best_prev_acc = None
                best_prev_version = None
                for v in prev_versions:
                    run_id_prev = getattr(v, 'run_id', None)
                    if not run_id_prev:
                        continue
                    try:
                        run_prev = client.get_run(run_id_prev)
                        acc_prev = run_prev.data.metrics.get('test_accuracy')
                        if acc_prev is not None and (best_prev_acc is None or acc_prev > best_prev_acc):
                            best_prev_acc = acc_prev
                            best_prev_version = v.version
                    except Exception:
                        continue
                if best_prev_acc is None:
                    print("  Não foi possível recuperar acurácia das versões anteriores. Não promovido.")
                else:
                    print(f"  Melhor acurácia anterior: {best_prev_acc:.4f} (versão {best_prev_version})")
                    if test_accuracy > best_prev_acc:
                        try:
                            client.set_registered_model_alias(
                                model_name,
                                "Production",
                                model_version.version
                            )
                            print(f"  ✓ Nova versão {model_version.version} promovida a Production (acurácia {test_accuracy:.4f} > {best_prev_acc:.4f}).")
                        except Exception as e:
                            print(f"  ⚠ Falha ao definir alias Production: {e}")
                    else:
                        print(f"  ✗ Acurácia {test_accuracy:.4f} não supera {best_prev_acc:.4f}. Mantém Production existente.")
        except Exception as e:
            print(f"  ⚠ Erro ao registrar/promover modelo: {e}")
    
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
