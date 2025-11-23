# Diagrama do Fluxo CI/CD - Aula 06

## 📊 Visão Geral do Pipeline Atualizado

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESENVOLVEDOR                                 │
│  - Ajusta hiperparâmetros diretamente no train.py               │
│    (comentando/descomentando blocos)                            │
│  - Modifica preprocessing/testes conforme necessário            │
│  - Executa testes locais antes do push                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ git commit & push
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB REPOSITORY                             │
│  - Detecta alterações na branch main                            │
│  - Trigger: paths em aula_06_cicd_automacao/**, data/**         │
│    ou no próprio workflow                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Dispara GitHub Actions
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              JOB 1: TESTES (test)                                │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Setup Python 3.10                              │          │
│  │ 3. Instalar dependências (requirements.txt)       │          │
│  │ 4. Executar: python test_pipeline.py             │          │
│  │                                                    │          │
│  │ Testes garantem:                                  │          │
│  │  ✓ Integridade dos transformers                   │          │
│  │  ✓ Pipeline sklearn consistente                   │          │
│  │  ✓ Qualidade mínima dos dados                     │          │
│  └───────────────────────────────────────────────────┘          │
│                         │                                        │
│                         │ Falhou → STOP ❌                       │
│                         │ Passou → Próximo ✅                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│       JOB 2: TREINAR MODELO (train-model)                  │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Setup Python 3.10                              │          │
│  │ 3. Instalar dependências                          │          │
│  │ 4. Executar: python train.py                      │          │
│  │                                                    │          │
│  │ Pipeline de Treinamento:                          │          │
│  │  ├─ Carrega heart_disease_uci.csv                │          │
│  │  ├─ train_test_split estratificado (80/20)       │          │
│  │  ├─ Pipeline sklearn                             │          │
│  │  │   • MissingValueImputer                       │          │
│  │  │   • CategoricalEncoder                        │          │
│  │  │   • FeatureEngineer                           │          │
│  │  │   • StandardScaler                            │          │
│  │  │   • RandomForestClassifier (params atuais)    │          │
│  │  ├─ Métricas: accuracy, precision, recall, f1    │          │
│  │  └─ Log no MLflow (params + métricas + modelo)   │          │
│  └───────────────────────────────────────────────────┘          │
│                         │                                        │
│                         │ Upload opcional do diretório usado no  │
│                         │ MLflow (mlruns/ local, mlruns_ci no CI)│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          JOB 3: RESUMO (summary)                                 │
│  ┌───────────────────────────────────────────────────┐          │
│  │ Exibe mensagens finais e lembra de verificar o    │          │
│  │ MLflow UI para visualizar runs e promoções.       │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo de Decisão Atual

```
┌────────────────┐
│  Commit/Push   │
└───────┬────────┘
        │
        ▼
┌────────────────┐      ❌ Falhou
│  Testes        ├──────────────────► STOP (corrigir e reenviar)
└───────┬────────┘
        │ ✅ Passou
        ▼
┌────────────────┐
│  Treinamento   │
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Sempre registra nova versão no       │
│ MLflow Model Registry                │
└───────┬──────────────────────────────┘
        │
        ├─ Primeira versão? → Sim
        │       │
        │       └► Define alias Production automaticamente
        │
        └─ Primeira versão? → Não
                │
                ▼
       Recupera melhor acurácia anterior
                │
                ├─ Nova acurácia > melhor anterior?
                │        │
                │        └► Atualiza alias Production para versão nova
                │
                └─ Caso contrário, mantém Production atual
```

---

## 📂 Artefatos Gerados

```
aula_06_cicd_automacao/
├── mlruns/ (execuções locais)
│   └── ...
├── mlruns_ci/ (execuções via GitHub Actions)
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── artifacts/model/        # Pipeline completo
│           ├── metrics/test_accuracy   # e demais métricas
│           ├── params/*.yaml           # hiperparâmetros ativos
│           └── tags/                   # metadados da execução
└── preprocessing.py                    # Referenciado no MLflow
```

---

## 🎯 Critérios de Registro e Promoção

| Situação | Ação |
|----------|------|
| Testes falham | ❌ Pipeline interrompido antes do treino |
| Treinamento roda (qualquer acurácia) | ✅ Run logado no MLflow e versão registrada |
| Não há versões anteriores | 🚀 Alias `Production` aponta para a versão recém-criada |
| Há versões anteriores e `test_accuracy` atual > melhor anterior | 🏆 Alias `Production` movido para a nova versão |
| Há versões anteriores e `test_accuracy` atual ≤ melhor anterior | ⚖️ Alias `Production` permanece na melhor versão anterior |

Observação: o script continua imprimindo alerta e retornando código de erro se a acurácia cair abaixo de 0.50, garantindo visibilidade quando algo muito errado ocorre.

---

## 🧠 Dicas para Ajustar Hiperparâmetros

- O bloco ativo de hiperparâmetros fica em `train.py` (variável `ACTIVE_PARAMS`).
- Para experimentar novos valores, comente/descomente o bloco desejado ou edite diretamente os parâmetros.
- Basta commitar a mudança e o pipeline executará com essa configuração na próxima execução.

---

## 📚 Boas Práticas Mantidas

✅ Separação clara entre preprocessing, treino e testes.  
✅ Testes unitários como gate antes do treinamento.  
✅ MLflow como fonte da verdade para métricas, parâmetros e modelos.  
✅ Registro automático de todas as execuções no Model Registry.  
✅ Promoção baseada em métrica objetiva (acurácia).  
✅ Workflow simples e declarativo no GitHub Actions.

---

Este documento acompanha o fluxo atual do pipeline da Aula 06, já com o modelo único e promoção automática guiada por métricas. 🚀
