"""
PHASE 1: Initial Linear Regression with Survey Data Only
=========================================================
Goal: Demonstrate that survey data alone gives poor predictions
for understanding junk food purchase behavior.

This is the baseline model. We expect a LOW R² which will motivate
the integration of INEGI socioeconomic data in Phase 2.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('outputs', exist_ok=True)

print("=" * 80)
print("PHASE 1: INITIAL LINEAR REGRESSION (Survey Data Only)")
print("=" * 80)

# ============================================================================
# 1. LOAD AND CLEAN DATA
# ============================================================================
tec = pd.read_csv('src/data/survey_responses.csv', sep=';', encoding='utf-8-sig')
forms = pd.read_csv('src/data/forms_survey_responses.csv', encoding='iso-8859-1')

print(f"\n[Data] Tec Survey:   {tec.shape[0]} responses")
print(f"[Data] Forms Survey: {forms.shape[0]} responses")

# ============================================================================
# 2. FEATURE ENGINEERING - Convert categorical to numeric
# ============================================================================

def encode_age(age):
    """Convert age range to midpoint."""
    if pd.isna(age):
        return np.nan
    age = str(age).strip()
    if 'Menor' in age or '<18' in age:
        return 16
    if '18-30' in age:
        return 24
    if '31-40' in age:
        return 35
    if '41-50' in age:
        return 45
    if 'Mayor' in age or '>50' in age:
        return 58
    return np.nan

def encode_cooking_time(val):
    """Cooking frequency to numeric (times/week)."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if 'Nunca' in val:
        return 0
    if '1-2' in val:
        return 1.5
    if '3-4' in val:
        return 3.5
    if 'Mas de 5' in val or '5+' in val:
        return 6
    return np.nan

def encode_junk_food(val):
    """Junk food purchase frequency (times/week as float)."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if 'Diario' in val:
        return 7.0
    if '2-3 veces a la semana' in val:
        return 2.5
    if 'Una vez a la semana' in val:
        return 1.0
    if 'Una vez al mes' in val:
        return 0.25
    if 'Nunca' in val:
        return 0
    return np.nan

def encode_willingness(val):
    """Willingness to change habits."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == 'Si':
        return 1.0
    if 'Tal vez' in val:
        return 0.5
    if val == 'No':
        return 0
    return np.nan

def encode_parent(val):
    if pd.isna(val):
        return 0
    return 1 if str(val).strip() == 'Si' else 0

def encode_agreement(val):
    """Likert scale: agreement with 'eat fast/cheap > nutritious'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    mapping = {
        'Muy en desacuerdo': 1,
        'En desacuerdo': 2,
        'Neutral': 3,
        'De acuerdo': 4,
        'Muy de acuerdo': 5,
    }
    return mapping.get(val, np.nan)

def encode_work_type(val):
    """Work category to ordinal."""
    if pd.isna(val):
        return 0
    val = str(val).lower()
    if 'estudia' in val or 'escuela' in val or 'universidad' in val:
        return 1
    if 'ningun' in val or 'no trabaj' in val or 'sin trabaj' in val:
        return 0
    if 'oficina' in val or 'administrat' in val or 'corporativo' in val:
        return 3
    if 'ama de casa' in val or 'hogar' in val:
        return 2
    return 2

# Helper to find column by keyword (handles weird unicode in column names)
def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

# Identify columns flexibly
def extract_features(df):
    age_col = find_col(df, 'edad')
    cook_col = find_col(df, 'cuantas', 'cocinar')
    work_col = find_col(df, 'tipo', 'trabajo')
    parent_col = find_col(df, 'eres', 'padre')
    will_col = find_col(df, 'dispuesto', 'cambiar')
    agree_col = find_col(df, 'acuerdo', 'saciar')
    junk_col = find_col(df, 'seguido', 'comida')

    return pd.DataFrame({
        'age': df[age_col].apply(encode_age) if age_col else np.nan,
        'cooking_freq': df[cook_col].apply(encode_cooking_time) if cook_col else np.nan,
        'work_type': df[work_col].apply(encode_work_type) if work_col else np.nan,
        'is_parent': df[parent_col].apply(encode_parent) if parent_col else 0,
        'willingness_change': df[will_col].apply(encode_willingness) if will_col else np.nan,
        'fast_cheap_pref': df[agree_col].apply(encode_agreement) if agree_col else np.nan,
        'junk_food_freq': df[junk_col].apply(encode_junk_food) if junk_col else np.nan,
    })

df = extract_features(tec)
forms_df = extract_features(forms)

# Combine
df = pd.concat([df, forms_df], ignore_index=True)
df_clean = df.dropna()

print(f"\n[Features] Total respondents: {len(df)}")
print(f"[Features] Complete cases (no NaN): {len(df_clean)}")
print(f"[Features] Variables: {list(df_clean.columns)}")

print(f"\n[Stats] Junk food frequency (times/week):")
print(df_clean['junk_food_freq'].describe())

# ============================================================================
# 3. INITIAL LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("LINEAR REGRESSION - SURVEY DATA ONLY")
print("=" * 80)

features = ['age', 'cooking_freq', 'work_type', 'is_parent',
            'willingness_change', 'fast_cheap_pref']
X = df_clean[features].values
y = df_clean['junk_food_freq'].values

# Standardize for interpretability of coefficients
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Fit
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

# Cross-validation
cv_scores = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring='r2')

print(f"\n[Metrics]")
print(f"  R² (Train):     {r2_train:.4f}")
print(f"  R² (Test):      {r2_test:.4f}")
print(f"  R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  RMSE (Test):    {rmse_test:.4f} times/week")
print(f"  MAE (Test):     {mae_test:.4f} times/week")

# Coefficients
print(f"\n[Coefficients - Standardized]")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_,
    'Abs_Importance': np.abs(model.coef_)
}).sort_values('Abs_Importance', ascending=False)
print(coef_df.to_string(index=False))

print(f"\n[Intercept] {model.intercept_:.4f}")

# ============================================================================
# 4. DIAGNOSIS - Why is the model bad?
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSIS - WHY IS THIS MODEL INSUFFICIENT?")
print("=" * 80)

print(f"""
The R² of {r2_test:.3f} means our model explains only {r2_test*100:.1f}% of variance.

PROBLEMS IDENTIFIED:
  1. Survey captures INDIVIDUAL behavior but not CONTEXT (where they live,
     what stores are nearby, their income level)

  2. Self-reported behavior is noisy (social desirability bias)

  3. Missing key variables:
     - SOCIOECONOMIC: Income, marginalization index
     - GEOGRAPHIC: Nanostore density, urbanization
     - HEALTH: Regional diabetes/obesity prevalence

  4. Linear relationship is too simple for behavioral data

WHAT WE NEED:
  → INEGI DENUE: Store density per state (geographic context)
  → INSP ENSANUT: Health prevalence (validation of impact)
  → INEGI ENIGH: Income/spending patterns (economic context)
  → CONAPO: Marginalization index (vulnerability context)

NEXT STEP (Phase 2):
  Integrate these datasets to build a Machine Learning model that captures
  the COMPLEX, NON-LINEAR relationships between individual behavior and
  socioeconomic context.
""")

# ============================================================================
# 5. SAVE RESULTS & VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolor='k')
axes[0].plot([0, 7], [0, 7], 'r--', label='Perfect prediction')
axes[0].set_xlabel('Actual Junk Food Frequency (times/week)', fontsize=11)
axes[0].set_ylabel('Predicted Frequency', fontsize=11)
axes[0].set_title(f'Linear Regression: Predicted vs Actual\nR² = {r2_test:.3f} (POOR FIT)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Coefficients bar chart
colors = ['#d62728' if c < 0 else '#2ca02c' for c in coef_df['Coefficient']]
axes[1].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='k')
axes[1].axvline(x=0, color='k', linewidth=0.5)
axes[1].set_xlabel('Standardized Coefficient', fontsize=11)
axes[1].set_title('Feature Influence on Junk Food Purchase', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/01_linear_regression_results.png', dpi=120, bbox_inches='tight')
print(f"[Saved] outputs/01_linear_regression_results.png")

# Save processed data for next phase
df_clean.to_csv('outputs/processed_survey_data.csv', index=False)
print(f"[Saved] outputs/processed_survey_data.csv ({len(df_clean)} rows)")

# Save metrics
metrics = {
    'phase': 'Phase 1: Initial Linear Regression',
    'n_samples': int(len(df_clean)),
    'n_features': len(features),
    'r2_train': float(r2_train),
    'r2_test': float(r2_test),
    'r2_cv_mean': float(cv_scores.mean()),
    'r2_cv_std': float(cv_scores.std()),
    'rmse': float(rmse_test),
    'mae': float(mae_test),
    'conclusion': 'POOR FIT - need INEGI socioeconomic data'
}

import json
with open('outputs/01_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"[Saved] outputs/01_metrics.json")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE - Continue with Phase 2: ML with INEGI Integration")
print("=" * 80)
