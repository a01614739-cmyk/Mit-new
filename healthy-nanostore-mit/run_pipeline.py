"""
Healthy Nanostore - Complete ML Pipeline Orchestrator
======================================================
Runs all 3 phases of the analysis pipeline in order.

Usage:
    python run_pipeline.py

Outputs (in outputs/):
    - 01_metrics.json, 01_linear_regression_results.png
    - 02_metrics.json, 02_ml_results.png, 02_feature_importance.csv
    - 03_meal_plans.json, 03_meal_plans.png, 03_cluster_profiles.csv
    - integrated_dataset.csv, processed_survey_data.csv
"""

import subprocess
import sys
import os
import time

phases = [
    ('Phase 1: Initial Linear Regression', 'src/ml/01_initial_regression.py'),
    ('Phase 2: ML with INEGI Integration', 'src/ml/02_ml_with_inegi.py'),
    ('Phase 3: Family Segmentation & Meal Plans', 'src/ml/03_meal_plan.py'),
]

print("=" * 80)
print("HEALTHY NANOSTORE - FULL PIPELINE")
print("=" * 80)

start_time = time.time()
results = []

for name, script in phases:
    print(f"\n🚀 Running: {name}")
    print(f"   Script: {script}")
    phase_start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True
    )

    duration = time.time() - phase_start

    if result.returncode == 0:
        status = "✅ SUCCESS"
        results.append((name, True, duration))
    else:
        status = "❌ FAILED"
        results.append((name, False, duration))
        print(f"   STDERR: {result.stderr[-500:]}")

    print(f"   {status} ({duration:.1f}s)")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
for name, success, duration in results:
    icon = "✅" if success else "❌"
    print(f"  {icon} {name:50s} ({duration:.1f}s)")
print(f"\n📊 Total runtime: {total_time:.1f}s")
print(f"📁 Outputs available in: outputs/")
print(f"🌐 Dashboard: src/web/results_dashboard.html")
