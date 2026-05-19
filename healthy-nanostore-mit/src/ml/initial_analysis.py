"""
Initial Exploratory Data Analysis - Healthy Nanostore
Analyzes both Tec survey and Google Forms survey
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load data
tec_survey = pd.read_csv('src/data/survey_responses.csv', sep=';', encoding='utf-8-sig')
forms_survey = pd.read_csv('src/data/forms_survey_responses.csv', encoding='iso-8859-1')

print("=" * 80)
print("INITIAL EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DATASET OVERVIEW
# ============================================================================
print("\n1. DATASET OVERVIEW\n")
print(f"Tec Survey:   {tec_survey.shape[0]} respondents, {tec_survey.shape[1]} columns")
print(f"Forms Survey: {forms_survey.shape[0]} respondents, {forms_survey.shape[1]} columns")
print(f"Total:        {tec_survey.shape[0] + forms_survey.shape[0]} respondents")

# ============================================================================
# 2. DEMOGRAPHIC PROFILE
# ============================================================================
print("\n2. DEMOGRAPHIC PROFILE\n")

# Age distribution
if 'Edad' in tec_survey.columns:
    print("Age Distribution (Tec Survey):")
    print(tec_survey['Edad'].value_counts().sort_index())

# ============================================================================
# 3. KEY INSIGHTS: TIME & JUNK FOOD PURCHASE
# ============================================================================
print("\n3. KEY INSIGHT: COOKING TIME vs. JUNK FOOD CONSUMPTION\n")

cooking_cols = [col for col in tec_survey.columns if 'cocinar' in col.lower()]
junk_cols = [col for col in tec_survey.columns if 'comida rápida' in col.lower() or 'procesada' in col.lower()]

if cooking_cols and junk_cols:
    cooking_col = cooking_cols[0]
    junk_col = junk_cols[0]

    print(f"Cooking frequency: {cooking_col}")
    print(tec_survey[cooking_col].value_counts())
    print(f"\nJunk food purchase: {junk_col}")
    print(tec_survey[junk_col].value_counts())

    # Cross-tabulation
    crosstab = pd.crosstab(tec_survey[cooking_col], tec_survey[junk_col], margins=True)
    print("\nCrosstab: Cooking Time vs. Junk Food Frequency")
    print(crosstab)

# ============================================================================
# 4. WILLINGNESS TO CHANGE BEHAVIOR
# ============================================================================
print("\n4. WILLINGNESS TO CHANGE & BARRIERS\n")

change_cols = [col for col in tec_survey.columns if 'cambiar' in col.lower() and 'dispuesto' in col.lower()]
barrier_cols = [col for col in tec_survey.columns if 'desmotiva' in col.lower()]

if change_cols:
    print(f"Willingness to change habits:")
    for col in change_cols[:1]:
        print(tec_survey[col].value_counts())

if barrier_cols:
    print(f"\nTop barriers to healthy eating:")
    for col in barrier_cols[:1]:
        print(tec_survey[col].value_counts().head(10))

# ============================================================================
# 5. PARENTS SPECIFIC QUESTIONS
# ============================================================================
print("\n5. SINGLE PARENT & FAMILY INSIGHTS (Forms Survey)\n")

parent_cols = [col for col in forms_survey.columns if 'padre' in col.lower() or 'soltero' in col.lower()]
children_cols = [col for col in forms_survey.columns if 'hijo' in col.lower()]

if 'Eres padre/madre?' in forms_survey.columns:
    print("Parental status:")
    print(forms_survey['Eres padre/madre?'].value_counts())

# ============================================================================
# 6. SOLUTIONS REQUESTED
# ============================================================================
print("\n6. MOST REQUESTED SOLUTIONS\n")

solutions_cols = [col for col in tec_survey.columns if 'soluciones' in col.lower()]
if solutions_cols:
    print(f"Solutions to improve habits:")
    sol_text = tec_survey[solutions_cols[0]].dropna().astype(str)
    print(f"Sample responses:\n{sol_text.head(5).to_string()}")

# ============================================================================
# 7. SUMMARY STATISTICS FOR REGRESSION
# ============================================================================
print("\n7. SUMMARY FOR LINEAR REGRESSION\n")

# Numeric columns only
numeric_cols = tec_survey.select_dtypes(include=[np.number]).columns
print(f"Numeric columns available: {list(numeric_cols)}")

# Try to identify key variables
print("\nKey Variables for Initial Regression:")
print(f"- Sample size: {tec_survey.shape[0]}")
print(f"- Missing values: {tec_survey.isnull().sum().sum()} cells")
print(f"- Complete cases: {tec_survey.dropna().shape[0]}")

# ============================================================================
# 8. NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. INITIAL LINEAR REGRESSION (current data):
   - Dependent: Junk food purchase frequency
   - Predictors: Cooking time, work hours, age, parenthood status
   - Expected: Poor R², highlighting need for additional data

2. INTEGRATE INEGI DATA:
   - DENUE: Geographic distribution of stores
   - ENSANUT: Regional health/nutrition prevalence
   - ENIGH: Income levels by region

3. MACHINE LEARNING PIPELINE:
   - Combine survey responses + INEGI socioeconomic variables
   - Feature engineering: income decile, urbanization, store density
   - Improved predictions: Why families buy junk food (price, access, time)

4. DIETARY INTERVENTION PLAN:
   - Segment families by income + household structure
   - Tailor meal plans for single parents, large families, low-income
   - Integration with nanostore supply chain

""")

print("=" * 80)
