"""
Módulo de pré-processamento para o pipeline de ML.
Baseado nos transformers customizados da Aula 04.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Imputa valores faltantes usando mediana (numéricas) e moda (categóricas)."""
    
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.num_medians_ = {}
        self.cat_modes_ = {}

    def fit(self, X, y=None):
        # Calcula mediana para colunas numéricas
        for col in self.numeric_cols:
            if col in X.columns:
                self.num_medians_[col] = X[col].median()
        
        # Calcula moda para colunas categóricas
        for col in self.categorical_cols:
            if col in X.columns:
                mode_series = X[col].mode()
                if len(mode_series) > 0:
                    self.cat_modes_[col] = mode_series[0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, val in self.num_medians_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(val)
        for col, val in self.cat_modes_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(val)
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Aplica mapeamentos categóricos e one-hot encoding com colunas estáveis."""
    
    def __init__(self):
        self.categorical_mappings = {
            'sex': {0: 'Female', 1: 'Male'},
            'cp': {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal', 3: 'asymptomatic'},
            'restecg': {0: 'normal', 1: 'st-t abnormality', 2: 'left ventricular hypertrophy'},
            'slope': {0: 'upsloping', 1: 'flat', 2: 'downsloping'},
            'thal': {0: 'normal', 1: 'fixed defect', 2: 'reversable defect'}
        }
        self.categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
        self.one_hot_columns_ = []

    def _apply_mappings(self, X):
        X_copy = X.copy()
        for col, mapping in self.categorical_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(
                    lambda x: mapping.get(int(x) if pd.api.types.is_numeric_dtype(X_copy[col]) else x, x)
                )
        return X_copy

    def fit(self, X, y=None):
        X_mapped = self._apply_mappings(X)
        cols_to_encode = [c for c in self.categorical_cols if c in X_mapped.columns]
        if cols_to_encode:
            X_encoded = pd.get_dummies(X_mapped, columns=cols_to_encode, drop_first=False)
        else:
            X_encoded = X_mapped
        self.one_hot_columns_ = X_encoded.columns.tolist()
        return self

    def transform(self, X):
        X_mapped = self._apply_mappings(X)
        cols_to_encode = [c for c in self.categorical_cols if c in X_mapped.columns]
        if cols_to_encode:
            X_encoded = pd.get_dummies(X_mapped, columns=cols_to_encode, drop_first=False)
        else:
            X_encoded = X_mapped.copy()

        # Adicionar colunas ausentes com zeros
        missing_cols = [c for c in self.one_hot_columns_ if c not in X_encoded.columns]
        for col in missing_cols:
            X_encoded[col] = 0

        # Remover colunas extras
        extra_cols = [c for c in X_encoded.columns if c not in self.one_hot_columns_]
        if extra_cols:
            X_encoded = X_encoded.drop(columns=extra_cols)

        # Reordenar conforme fit
        X_encoded = X_encoded[self.one_hot_columns_]
        return X_encoded


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria features derivadas (age_squared, bp_chol_ratio, etc.)"""
    
    def __init__(self):
        self.eps = 1
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Age features
        if 'age' in X_copy.columns:
            X_copy['age_squared'] = X_copy['age'] ** 2
            X_copy['age_decade'] = (X_copy['age'] // 10).astype(int)
        
        # Cholesterol to age ratio
        if 'chol' in X_copy.columns and 'age' in X_copy.columns:
            X_copy['cholesterol_to_age'] = X_copy['chol'] / (X_copy['age'] + self.eps)
        
        # Max heart rate percentage
        if 'thalch' in X_copy.columns and 'age' in X_copy.columns:
            predicted_max_hr = (220 - X_copy['age']).clip(lower=1)
            X_copy['max_hr_pct'] = X_copy['thalch'] / (predicted_max_hr + self.eps)
        
        # Blood pressure to cholesterol ratio
        if 'trestbps' in X_copy.columns and 'chol' in X_copy.columns:
            X_copy['bp_chol_ratio'] = X_copy['trestbps'] / (X_copy['chol'] + self.eps)
        
        # Binary flags
        if 'fbs' in X_copy.columns:
            X_copy['fbs_flag'] = X_copy['fbs'].astype(int)
        if 'exang' in X_copy.columns:
            X_copy['exang_flag'] = X_copy['exang'].astype(int)
        
        # Stress index
        if 'thalch' in X_copy.columns and 'trestbps' in X_copy.columns:
            X_copy['stress_index'] = X_copy['thalch'] / (X_copy['trestbps'] + self.eps)
        
        # Risk interaction
        if 'age' in X_copy.columns and 'oldpeak' in X_copy.columns:
            X_copy['risk_interaction'] = X_copy['age'] * X_copy['oldpeak']
        
        # High ST depression flag
        if 'oldpeak' in X_copy.columns:
            X_copy['high_st_depression_flag'] = (X_copy['oldpeak'] > 1.0).astype(int)
        
        return X_copy
