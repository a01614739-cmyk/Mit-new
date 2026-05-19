"""
PHASE 3: Family Segmentation & Personalized Meal Plan
======================================================
Goal: Use clustering to identify family types and create tailored
healthy meal plans optimized for Mexican nanostore (tiendita) supply.

Outputs:
- Family clusters (K-means)
- Cluster profiles
- Meal plans per cluster (cost, prep time, nutritional balance)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
np.random.seed(42)

print("=" * 80)
print("PHASE 3: FAMILY SEGMENTATION & MEAL PLAN GENERATION")
print("=" * 80)

# Load integrated data
df = pd.read_csv('outputs/integrated_dataset.csv')
print(f"\n[Data] Integrated dataset: {len(df)} respondents")

# ============================================================================
# 1. FAMILY SEGMENTATION (K-means)
# ============================================================================
clustering_features = [
    'is_parent', 'cooking_freq', 'age',
    'Ingreso_Trimestral_MXN', 'Indice_Marginacion_Raw',
    'Densidad_por_1000', 'Pct_Urbano'
]

X = df[clustering_features].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K via elbow method
inertias = []
ks = range(2, 8)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Choose K=4 (good balance)
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_cluster = df.dropna(subset=clustering_features).copy()
df_cluster['cluster'] = clusters

print(f"\n[Clustering] K-means with K={K}")
print(f"[Clusters] Distribution:")
print(df_cluster['cluster'].value_counts().sort_index())

# ============================================================================
# 2. CLUSTER PROFILING
# ============================================================================
print("\n" + "=" * 80)
print("CLUSTER PROFILES")
print("=" * 80)

profile = df_cluster.groupby('cluster').agg({
    'age': 'mean',
    'is_parent': 'mean',
    'cooking_freq': 'mean',
    'junk_food_freq': 'mean',
    'Ingreso_Trimestral_MXN': 'mean',
    'Indice_Marginacion_Raw': 'mean',
    'Densidad_por_1000': 'mean',
    'Pct_Urbano': 'mean',
    'Tasa_Pobreza_Pct': 'mean'
}).round(2)

print(profile.to_string())

# Assign meaningful names based on profile
cluster_profiles = {}
for c in range(K):
    sub = df_cluster[df_cluster['cluster'] == c]
    avg_income = sub['Ingreso_Trimestral_MXN'].mean()
    avg_marg = sub['Indice_Marginacion_Raw'].mean()
    parent_rate = sub['is_parent'].mean()
    cooking = sub['cooking_freq'].mean()
    age = sub['age'].mean()

    # Build descriptive name
    if avg_income > 80000 and parent_rate < 0.3:
        name = "Joven Urbano Profesional"
        target = "young_urban"
    elif avg_income > 80000 and parent_rate >= 0.3:
        name = "Familia Urbana Acomodada"
        target = "affluent_family"
    elif avg_marg > 0.3:
        name = "Familia Vulnerable Alta Marginación"
        target = "vulnerable_family"
    elif parent_rate >= 0.3 and cooking < 3:
        name = "Familia Trabajadora Tiempo-Limitado"
        target = "busy_working_family"
    else:
        name = "Estudiante / Joven Recursos Medios"
        target = "student_youth"

    cluster_profiles[c] = {
        'name': name,
        'target_key': target,
        'size': len(sub),
        'avg_age': float(age),
        'parent_rate': float(parent_rate),
        'avg_income': float(avg_income),
        'cooking_freq': float(cooking),
        'junk_food_freq': float(sub['junk_food_freq'].mean()),
        'marginalization': float(avg_marg),
    }

for c, prof in cluster_profiles.items():
    print(f"\n[Cluster {c}] {prof['name']}")
    print(f"  Tamaño: {prof['size']} personas ({prof['size']/len(df_cluster)*100:.0f}%)")
    print(f"  Edad promedio: {prof['avg_age']:.0f} años")
    print(f"  % Padres/Madres: {prof['parent_rate']*100:.0f}%")
    print(f"  Ingreso trimestral: ${prof['avg_income']:,.0f} MXN")
    print(f"  Cocina/semana: {prof['cooking_freq']:.1f} veces")
    print(f"  Comida chatarra/sem: {prof['junk_food_freq']:.1f} veces")

# ============================================================================
# 3. MEAL PLANS PER CLUSTER
# ============================================================================
print("\n" + "=" * 80)
print("PERSONALIZED MEAL PLANS")
print("=" * 80)

# Meal database: products available in tienditas + healthy combinations
MEAL_DATABASE = {
    'young_urban': {
        'description': 'Plan económico de preparación rápida',
        'budget_weekly_mxn': 600,
        'prep_time_avg_min': 15,
        'meals': [
            {'day': 'Lun', 'desayuno': 'Avena con plátano y canela', 'comida': 'Sandwich integral de atún + ensalada', 'cena': 'Yogurt con granola y fruta'},
            {'day': 'Mar', 'desayuno': 'Huevos revueltos con frijoles', 'comida': 'Quesadillas con nopales y queso panela', 'cena': 'Sopa de verduras + tostadas'},
            {'day': 'Mie', 'desayuno': 'Smoothie de papaya y avena', 'comida': 'Atún con galletas integrales + jitomate', 'cena': 'Tortas de aguacate con frijoles'},
            {'day': 'Jue', 'desayuno': 'Pan integral con frijoles y aguacate', 'comida': 'Ensalada de pollo con espinaca', 'cena': 'Tacos de zanahoria rayada y queso'},
            {'day': 'Vie', 'desayuno': 'Avena con manzana y nuez', 'comida': 'Sopa instantánea+ verduras al vapor', 'cena': 'Quesadillas de espinaca'},
        ],
        'tiendita_shopping_list': ['Avena', 'Huevos', 'Frijoles', 'Atún en lata', 'Tortillas', 'Yogurt', 'Plátano', 'Manzana', 'Pan integral', 'Aguacate'],
        'nutrition_focus': 'Proteína vegetal, fibra, antioxidantes',
    },
    'affluent_family': {
        'description': 'Plan balanceado para familia con tiempo y recursos',
        'budget_weekly_mxn': 2500,
        'prep_time_avg_min': 35,
        'meals': [
            {'day': 'Lun', 'desayuno': 'Avena con frutos rojos y semillas', 'comida': 'Pollo asado + arroz integral + brócoli', 'cena': 'Ensalada César con pollo'},
            {'day': 'Mar', 'desayuno': 'Huevos con espinaca y queso fresco', 'comida': 'Pescado al horno + ensalada mediterránea', 'cena': 'Sopa de lentejas + pan integral'},
            {'day': 'Mie', 'desayuno': 'Smoothie verde con proteína', 'comida': 'Tacos de pescado con guacamole', 'cena': 'Hamburguesas vegetarianas'},
            {'day': 'Jue', 'desayuno': 'Pan integral con aguacate y huevo', 'comida': 'Pollo a la plancha + quinoa + verduras', 'cena': 'Pasta integral con verduras'},
            {'day': 'Vie', 'desayuno': 'Yogurt griego con miel y frutas', 'comida': 'Salmón + arroz salvaje + espárragos', 'cena': 'Sopa de calabaza + pan'},
        ],
        'tiendita_shopping_list': ['Avena', 'Huevos', 'Pollo', 'Pescado', 'Brócoli', 'Espinaca', 'Aguacate', 'Quinoa', 'Lentejas', 'Yogurt griego', 'Tomates'],
        'nutrition_focus': 'Variedad, omega-3, proteína magra, fibra',
    },
    'vulnerable_family': {
        'description': 'Plan económico-nutritivo para familias de alta marginación',
        'budget_weekly_mxn': 450,
        'prep_time_avg_min': 25,
        'meals': [
            {'day': 'Lun', 'desayuno': 'Atole de avena con plátano', 'comida': 'Arroz con frijoles + huevo', 'cena': 'Sopa de fideo + tortillas'},
            {'day': 'Mar', 'desayuno': 'Tortilla con frijoles y queso', 'comida': 'Sopa de lentejas + arroz', 'cena': 'Quesadillas con verduras'},
            {'day': 'Mie', 'desayuno': 'Avena con canela y piloncillo', 'comida': 'Calabacitas con pollo + arroz', 'cena': 'Tostadas de frijol + ensalada'},
            {'day': 'Jue', 'desayuno': 'Huevos con frijoles', 'comida': 'Sopa de verduras + tortillas + huevo', 'cena': 'Quesadillas de papa con salsa verde'},
            {'day': 'Vie', 'desayuno': 'Pan con frijoles refritos', 'comida': 'Arroz a la mexicana + huevo', 'cena': 'Tacos de frijol con nopales'},
        ],
        'tiendita_shopping_list': ['Frijoles', 'Arroz', 'Huevos', 'Tortillas', 'Avena', 'Lentejas', 'Verduras de temporada', 'Calabacitas', 'Nopales', 'Cebolla'],
        'nutrition_focus': 'Proteína vegetal abundante, granos enteros, costos optimizados',
        'subsidios': 'Compatible con apoyos de Liconsa, Diconsa, Bienestar',
    },
    'busy_working_family': {
        'description': 'Plan rápido para padres/madres con poco tiempo',
        'budget_weekly_mxn': 1200,
        'prep_time_avg_min': 20,
        'meals': [
            {'day': 'Lun', 'desayuno': 'Smoothie de plátano + avena', 'comida': 'Pollo a la plancha + arroz + ensalada', 'cena': 'Quesadillas con espinaca'},
            {'day': 'Mar', 'desayuno': 'Huevos rápidos con frijoles', 'comida': 'Tacos de pollo (preparado domingo)', 'cena': 'Sopa de fideos con verduras'},
            {'day': 'Mie', 'desayuno': 'Pan integral con queso y aguacate', 'comida': 'Sopa de pollo (olla express)', 'cena': 'Sandwich + ensalada'},
            {'day': 'Jue', 'desayuno': 'Yogurt con granola y fruta', 'comida': 'Pollo rostizado + arroz + verduras', 'cena': 'Quesadillas con jamón y verduras'},
            {'day': 'Vie', 'desayuno': 'Avena instantánea + fruta', 'comida': 'Pizza casera con verduras', 'cena': 'Sopa de calabaza + pan'},
        ],
        'tiendita_shopping_list': ['Pollo', 'Tortillas', 'Yogurt', 'Frijoles', 'Huevos', 'Verduras pre-lavadas', 'Avena instantánea', 'Pan integral'],
        'nutrition_focus': 'Conveniencia + nutrición, batch cooking, snacks saludables',
        'meal_prep_strategy': 'Domingo: cocer 2kg de pollo, picar verduras, hacer salsas',
    },
    'student_youth': {
        'description': 'Plan estudiante con bajo costo y fácil preparación',
        'budget_weekly_mxn': 500,
        'prep_time_avg_min': 12,
        'meals': [
            {'day': 'Lun', 'desayuno': 'Avena con fruta', 'comida': 'Sandwich + yogurt', 'cena': 'Quesadillas con frijoles'},
            {'day': 'Mar', 'desayuno': 'Huevos con tortilla', 'comida': 'Sopa instantánea + huevo + verduras', 'cena': 'Tostadas con frijoles y aguacate'},
            {'day': 'Mie', 'desayuno': 'Smoothie de plátano + leche', 'comida': 'Atún con galletas + zanahoria', 'cena': 'Pasta con salsa de tomate'},
            {'day': 'Jue', 'desayuno': 'Cereal integral con leche', 'comida': 'Arroz con huevo y ensalada', 'cena': 'Quesadillas + ensalada'},
            {'day': 'Vie', 'desayuno': 'Pan con frijoles y queso', 'comida': 'Sopa de pollo + arroz', 'cena': 'Sandwich integral + fruta'},
        ],
        'tiendita_shopping_list': ['Avena', 'Pan integral', 'Huevos', 'Frijoles', 'Tortillas', 'Yogurt', 'Plátano', 'Atún', 'Leche'],
        'nutrition_focus': 'Energía sostenida, proteína suficiente para estudio',
    }
}

# Print meal plans per cluster
meal_plans_output = {}
for c, prof in cluster_profiles.items():
    plan = MEAL_DATABASE.get(prof['target_key'], MEAL_DATABASE['young_urban'])

    print(f"\n{'─'*80}")
    print(f"CLUSTER {c}: {prof['name']}")
    print(f"{'─'*80}")
    print(f"📋 {plan['description']}")
    print(f"💰 Presupuesto semanal: ${plan['budget_weekly_mxn']:,} MXN")
    print(f"⏱️  Tiempo promedio prep: {plan['prep_time_avg_min']} min/comida")
    print(f"🎯 Enfoque nutricional: {plan['nutrition_focus']}")
    print(f"\n📅 Plan semanal:")

    for meal in plan['meals']:
        print(f"  {meal['day']}: 🌅 {meal['desayuno']}")
        print(f"        🍽️  {meal['comida']}")
        print(f"        🌙 {meal['cena']}")

    print(f"\n🛒 Lista para tienditas:")
    for item in plan['tiendita_shopping_list']:
        print(f"   • {item}")

    if 'subsidios' in plan:
        print(f"\n💡 {plan['subsidios']}")
    if 'meal_prep_strategy' in plan:
        print(f"\n📌 Estrategia: {plan['meal_prep_strategy']}")

    meal_plans_output[f"cluster_{c}_{prof['name'].replace(' ', '_')}"] = {
        'profile': prof,
        'meal_plan': plan
    }

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n[Generating visualizations...]")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Elbow method
ax1 = plt.subplot(2, 3, 1)
ax1.plot(ks, inertias, 'bo-', linewidth=2, markersize=10)
ax1.axvline(x=K, color='r', linestyle='--', label=f'Chosen K={K}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
ax1.set_ylabel('Inertia', fontsize=11)
ax1.set_title('Elbow Method - Optimal K', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: PCA 2D visualization of clusters
ax2 = plt.subplot(2, 3, 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for c in range(K):
    mask = clusters == c
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c],
                label=cluster_profiles[c]['name'][:30],
                alpha=0.7, edgecolor='k', s=60)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
ax2.set_title('Family Segments (PCA 2D)', fontsize=12)
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Cluster size
ax3 = plt.subplot(2, 3, 3)
sizes = [cluster_profiles[c]['size'] for c in range(K)]
names = [cluster_profiles[c]['name'] for c in range(K)]
ax3.barh(names, sizes, color=colors[:K], edgecolor='k')
ax3.set_xlabel('Number of Respondents', fontsize=11)
ax3.set_title('Cluster Sizes', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Budget vs Junk food consumption
ax4 = plt.subplot(2, 3, 4)
budgets = [MEAL_DATABASE[cluster_profiles[c]['target_key']]['budget_weekly_mxn'] for c in range(K)]
junk = [cluster_profiles[c]['junk_food_freq'] for c in range(K)]
for c in range(K):
    ax4.scatter(budgets[c], junk[c], c=colors[c], s=300,
                edgecolor='k', label=cluster_profiles[c]['name'][:25])
ax4.set_xlabel('Weekly Meal Budget (MXN)', fontsize=11)
ax4.set_ylabel('Current Junk Food Frequency (times/week)', fontsize=11)
ax4.set_title('Meal Budget vs. Current Junk Food Habit', fontsize=12)
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.3)

# Plot 5: Income comparison
ax5 = plt.subplot(2, 3, 5)
incomes = [cluster_profiles[c]['avg_income']/1000 for c in range(K)]
ax5.bar(range(K), incomes, color=colors[:K], edgecolor='k')
ax5.set_xticks(range(K))
ax5.set_xticklabels([cluster_profiles[c]['name'].split()[0] for c in range(K)], rotation=15)
ax5.set_ylabel('Quarterly Income (thousand MXN)', fontsize=11)
ax5.set_title('Income Profile by Segment', fontsize=12)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Prep time vs cooking frequency
ax6 = plt.subplot(2, 3, 6)
prep_times = [MEAL_DATABASE[cluster_profiles[c]['target_key']]['prep_time_avg_min'] for c in range(K)]
cooking = [cluster_profiles[c]['cooking_freq'] for c in range(K)]
for c in range(K):
    ax6.scatter(cooking[c], prep_times[c], c=colors[c], s=300,
                edgecolor='k', label=cluster_profiles[c]['name'][:25])
ax6.set_xlabel('Current Cooking Frequency (times/week)', fontsize=11)
ax6.set_ylabel('Recommended Prep Time (min/meal)', fontsize=11)
ax6.set_title('Cooking Capacity vs. Plan Complexity', fontsize=12)
ax6.legend(fontsize=8, loc='best')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/03_meal_plans.png', dpi=120, bbox_inches='tight')
print(f"[Saved] outputs/03_meal_plans.png")

# ============================================================================
# 5. SAVE OUTPUTS
# ============================================================================
with open('outputs/03_meal_plans.json', 'w', encoding='utf-8') as f:
    json.dump({
        'phase': 'Phase 3: Family Segmentation & Meal Plans',
        'n_clusters': K,
        'n_respondents': len(df_cluster),
        'clusters': meal_plans_output
    }, f, indent=2, ensure_ascii=False, default=str)

profile.to_csv('outputs/03_cluster_profiles.csv')

print(f"[Saved] outputs/03_meal_plans.json")
print(f"[Saved] outputs/03_cluster_profiles.csv")

print("\n" + "=" * 80)
print(f"PHASE 3 COMPLETE - {K} family segments with personalized meal plans")
print("=" * 80)
