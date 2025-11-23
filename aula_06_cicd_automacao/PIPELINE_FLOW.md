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
│                         │ MLflow (mlruns/ local,                 │
│                         │ mlruns_ci_snapshot no CI)              │
│                         │ Snapshot commitado p/ repo (usa        │
│                         │ permissão contents:write do workflow)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          JOB 3: REGISTRAR E PROMOVER (register-model)            │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout + setup Python                        │          │
│  │ 2. Baixa artifact ``mlruns`` da etapa anterior    │          │
│  │ 3. Executa: ``python register_model.py``          │          │
│  │                                                   │          │
│  │ O script lê ``latest_run.json`` e:                │          │
│  │  • Registra uma nova versão no Model Registry     │          │
│  │  • Compara ``test_accuracy`` com versões antigas  │          │
│  │  • Limpa versões órfãs e só move o alias          │          │
│  │    ``Production`` se houver ganho real            │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│          JOB 4: RESUMO (summary)                                 │
│  ┌───────────────────────────────────────────────────┐          │
│  │ Exibe mensagens finais e lembra de verificar o    │          │
│  │ MLflow UI para visualizar runs e promoções.       │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

> 💡 No GitHub Actions definimos `MLFLOW_TRACKING_FOLDER=mlruns_ci_snapshot`,
o que garante que cada nova execução reutilize o mesmo diretório de tracking,
mantendo o histórico e permitindo comparação com versões anteriores.

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
│ Salva latest_run.json com run_id,    │
│ model_uri, métricas                  │
└───────┬──────────────────────────────┘
        │
        ▼
┌────────────────┐
│  Registro      │
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Carrega best accuracy anterior       │
│ (via MLflowClient)                   │
└───────┬──────────────────────────────┘
        │
        ├─ Primeira versão? → Sim
        │       │
        │       └► Alias Production ← versão atual
        │
        └─ Primeira versão? → Não
                │
                ▼
       Nova acurácia > melhor anterior?
                │
                ├─ Sim → Atualiza alias Production
                └─ Não → Mantém versão anterior
```

---

## 📂 Artefatos Gerados

```
aula_06_cicd_automacao/
├── mlruns/ (execuções locais)
│   └── ...
├── mlruns_ci_snapshot/ (execuções via GitHub Actions e commitado)
│   ├── <experiment_id>/
│   │   └── <run_id>/
│   │       ├── artifacts/model/        # Pipeline completo
│   │       ├── metrics/test_accuracy   # e demais métricas
│   │       ├── params/*.yaml           # hiperparâmetros ativos
│   │       └── tags/                   # metadados da execução
│   └── latest_run.json                 # ponte entre treino e registro
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

### ▶️ Execução manual (fora do CI)

```
cd aula_06_cicd_automacao
python train.py            # loga o run e atualiza latest_run.json
python register_model.py   # registra e (se aplicável) promove
```

Mantendo `MLFLOW_TRACKING_FOLDER` apontando para o mesmo diretório, você garante
que o registro manual reflita exatamente o que o pipeline faria.

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
