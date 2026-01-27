# Aula 6: CI/CD e Automação de Pipelines

## Descrição
Automatize o ciclo de vida de modelos com pipelines de CI/CD usando GitHub Actions e MLflow.

## Objetivos
- Criar pipelines de ML automatizados
- Implementar versionamento de dados e código
- Automatizar testes de modelos
- Integrar MLflow com CI/CD

---

## 📚 Conteúdo da Aula

Esta aula apresenta um exemplo didático e completo de CI/CD para Machine Learning. A cada push na branch `main`, o pipeline automaticamente:

1. ✅ **Roda testes** no código de treinamento e pré-processamento
2. 🚀 **Treina modelos** (baseline ou otimizado)
3. 📊 **Avalia e loga** métricas no MLflow
4. 🏆 **Registra versões** no Model Registry se a acurácia for satisfatória

---

## 🗂️ Estrutura dos Arquivos

```
aula_06_cicd_automacao/
├── preprocessing.py       # Transformers customizados (Imputer, Encoder, FeatureEngineer)
├── train.py              # Script de treinamento
├── test_pipeline.py      # Suite de testes unitários
├── README.md            # Este arquivo
└── test.ipynb           # Notebook de exploração (opcional)

.github/workflows/
└── cicd-pipeline.yml    # GitHub Actions workflow
```

---

## 🚀 Como Usar

### 1️⃣ Executar Localmente

#### Rodar os testes:
```bash
cd aula_06_cicd_automacao
python test_pipeline.py
```

#### Treinar modelo (baseline por padrão):
```bash
python train.py --min-accuracy 0.75
```

#### Trocar para versão otimizada:
Edite `train.py`, descomente o bloco "VERSÃO OTIMIZADA" e comente a linha `ACTIVE_PARAMS = BASELINE_PARAMS`, depois execute:
```bash
python train.py --min-accuracy 0.75
```

#### Visualizar resultados no MLflow:
```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

### 2️⃣ Pipeline Automático (GitHub Actions)

O workflow é acionado automaticamente quando:
- Há **push na branch `main`** que modifica arquivos em `aula_06_cicd_automacao/` ou `data/`
- Ou via **execução manual** no GitHub (workflow_dispatch)

#### Passos do Pipeline:

**Job 1: Testes**
- Instala dependências
- Roda `test_pipeline.py`
- Valida transformers e funções

**Job 2: Treinar Modelo (ACTIVE_PARAMS)**
- Usa os hiperparâmetros atualmente ativos em `train.py`
- Loga no MLflow
- Registra no Model Registry se acurácia >= 0.75

**Job 3: Re-treinar após alteração para otimizado** (condicional)
- Executa se houver `[train-optimized]` no commit message (indicando que você editou para a versão otimizada)
- Usa hiperparâmetros que foram ativados ao descomentar o bloco em `train.py`
- Promove para "Production" se melhor que baseline

---

## 🎯 Exemplo Prático para Alunos

### Cenário 1: Testar o Baseline (padrão)
```bash
# (Opcional) Ajuste hiperparâmetros baseline em BASELINE_PARAMS
git add aula_06_cicd_automacao/train.py
git commit -m "Ajuste no modelo (baseline)"
git push origin main
```
→ Pipeline executa: testes + treino baseline

### Cenário 2: Treinar Versão Otimizada
1. Edite `train.py` e descomente o bloco "VERSÃO OTIMIZADA".
2. Comente a linha `ACTIVE_PARAMS = BASELINE_PARAMS`.
```bash
git add aula_06_cicd_automacao/train.py
git commit -m "[train-optimized] Ativa hiperparâmetros otimizados"
git push origin main
```
→ Pipeline executa: testes + treino otimizado (job condicional)

### Cenário 3: Execução Manual
No GitHub:
1. Vá para **Actions** → **CI/CD Pipeline - Heart Disease Model**
2. Clique em **Run workflow** (não há mais seleção de modelo, usa o código atual)

---

## 📋 Hiperparâmetros dos Modelos

### Baseline (Configuração Padrão - ACTIVE_PARAMS)
```python
{
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42
}
```

### Optimized (Descomentar bloco em `train.py`)
```python
{
    'n_estimators': 124,
    'max_depth': 15,
    'max_features': 2,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'criterion': 'gini',
    'random_state': 42
}
```

---

## 🧪 Testes Implementados

O `test_pipeline.py` valida:

✅ **Transformers**
- `MissingValueImputer`: imputação de valores faltantes
- `CategoricalEncoder`: encoding com colunas estáveis
- `FeatureEngineer`: criação de features derivadas
- Consistência do pipeline (mesmas colunas em treino/inferência)

✅ **Funções de Treinamento**
- Carregamento e preparação dos dados
- Criação do pipeline completo
- Hiperparâmetros baseline

✅ **Qualidade dos Dados**
- Existência do dataset
- Formato e colunas esperadas

---

## 📊 MLflow Integration

### Experimentos Logados
- **Experiment Name**: `heart-disease-cicd`
- **Run Names**: `baseline_random_forest`, `optimized_random_forest`

### Métricas Registradas
- `train_accuracy`
- `test_accuracy`
- `test_precision`
- `test_recall`
- `test_f1`

### Model Registry
- **Nome do Modelo**: `heart-disease-model`
- **Alias "Production"**: atribuído ao modelo otimizado se acurácia >= 0.75

---

## 🔧 Configuração (Opcional)

### MLflow Tracking Remoto

Para usar um servidor MLflow remoto (ex: Databricks), configure o secret no GitHub:

1. Vá em **Settings** → **Secrets and variables** → **Actions**
2. Adicione o secret `MLFLOW_TRACKING_URI` com a URL do servidor
3. Exemplo: `https://seu-mlflow-server.com`

Se não configurado, usa MLflow local (`file:./mlruns`)

---

## 💡 Conceitos Aprendidos

✅ **CI/CD para ML**: automação de testes, treino e deploy  
✅ **GitHub Actions**: workflows declarativos  
✅ **MLflow**: tracking, registry e versionamento  
✅ **Testes Automatizados**: validação contínua do código  
✅ **Model Registry**: gestão de versões e promoção  
✅ **Pipeline Sklearn**: encapsulamento de pré-processamento  
