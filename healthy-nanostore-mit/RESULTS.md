# 📊 Resultados del Análisis - Healthy Nanostore

## 🎯 Resumen Ejecutivo

Análisis end-to-end en 3 fases que demuestra cómo la integración de datos socioeconómicos del INEGI transforma un modelo predictivo deficiente en uno excelente, y se aplica para generar planes alimenticios personalizados para familias mexicanas.

| Fase | Modelo | R² Test | Conclusión |
|------|--------|---------|------------|
| **1. Solo Encuestas** | Linear Regression | **-1.16** | ❌ Insuficiente |
| **2. + INEGI** | Linear Regression | **0.910** | ✅ Excelente |
| **3. Segmentación** | K-Means (K=4) | — | ✅ 4 perfiles familiares |

**Mejora total:** +2.03 puntos de R² al integrar contexto socioeconómico.

---

## 📈 Fase 1: Regresión Lineal Inicial

### Datos
- **196 respuestas** (98 Tec + 98 Google Forms)
- **6 features individuales:** edad, frecuencia de cocina, tipo de trabajo, paternidad, disposición al cambio, preferencia rápido/barato

### Resultados
| Métrica | Valor |
|---------|-------|
| R² Train | 0.156 |
| R² Test | **-1.161** |
| R² CV (5-fold) | -0.057 ± 0.116 |
| RMSE | 2.73 veces/sem |
| MAE | 1.90 veces/sem |

### Diagnóstico
- R² negativo indica que el modelo **predice peor que la media**
- Las variables individuales no capturan suficiente varianza
- Se requiere contexto socioeconómico

---

## 🤖 Fase 2: Machine Learning con INEGI

### Datasets Integrados
| Fuente | Variable | Función |
|--------|----------|---------|
| **INEGI DENUE 2024** | Densidad de tiendas/1000 hab | Geografía retail |
| **INSP ENSANUT 2023** | Diabetes, obesidad, sobrepeso | Salud regional |
| **INEGI ENIGH 2022** | Ingreso, % gasto alimentos | Capacidad económica |
| **CONAPO 2020** | Índice marginación, % urbano, tasa pobreza | Vulnerabilidad social |

### Comparación de Modelos
| Modelo | R² Test | R² CV | RMSE | MAE |
|--------|---------|-------|------|-----|
| **Linear Regression** | **0.910** | 0.924 ± 0.030 | 0.50 | 0.38 |
| Ridge Regression | 0.909 | 0.927 ± 0.025 | 0.51 | 0.39 |
| Random Forest | 0.890 | 0.910 ± 0.023 | 0.56 | 0.46 |
| Gradient Boosting | 0.867 | 0.914 ± 0.019 | 0.61 | 0.50 |

### Importancia de Variables (Random Forest)

**Por tipo:**
- 🏘️ **Socioeconómicas:** 92.3% del poder predictivo
- 👤 **Individuales:** 7.7%

**Top 5 Variables:**
| Rank | Variable | Importancia | Tipo |
|------|----------|-------------|------|
| 1 | Ingreso Promedio Trimestral | 46.0% | Socioeconómico |
| 2 | % Gasto en Alimentos | 15.3% | Socioeconómico |
| 3 | Índice de Marginación | 13.9% | Socioeconómico |
| 4 | Densidad de Tiendas | 8.4% | Socioeconómico |
| 5 | % Población Urbana | 5.8% | Socioeconómico |

### Insights Clave
1. **Ingreso es el predictor #1** - las familias de menor ingreso compran significativamente más comida procesada
2. **Densidad de tiendas (DENUE) importa** - más nanostores cercanos → más compra impulsiva
3. **Marginación correlaciona con junk food** - validando la hipótesis ENSANUT
4. **Variables individuales son secundarias** - el contexto socioeconómico domina

---

## 🍽️ Fase 3: Segmentación Familiar + Plan Alimenticio

### Clusters Identificados (K-Means, K=4)

#### 🎓 Cluster 1: Estudiante / Joven Recursos Medios
- **Perfil:** ~22 años, ingreso $55K/trim, cocina 2 veces/sem
- **Plan:** $500 MXN/semana · 12 min/comida
- **Lista tiendita:** Avena, pan, huevos, frijoles, atún, plátano
- **Enfoque:** Energía sostenida, proteína para estudio

#### 👨‍💼 Cluster 2: Joven Urbano Profesional
- **Perfil:** ~25 años, ingreso $95K/trim, cocina 3 veces/sem
- **Plan:** $600 MXN/semana · 15 min/comida
- **Lista tiendita:** Avena, huevos, atún, aguacate, frijoles, yogurt
- **Enfoque:** Proteína vegetal, fibra, antioxidantes

#### 👨‍👩‍👧 Cluster 3: Familia Trabajadora Tiempo-Limitado
- **Perfil:** ~35 años, 80% padres/madres, ingreso $70K/trim
- **Plan:** $1,200 MXN/semana · 20 min/comida
- **Estrategia:** Batch cooking dominical (2kg pollo, picar verduras)
- **Lista tiendita:** Pollo, tortillas, frijoles, yogurt, verduras

#### 🏘️ Cluster 4: Familia Vulnerable Alta Marginación
- **Perfil:** ~32 años, ingreso $45K/trim, marginación alta (+0.8)
- **Plan:** $450 MXN/semana · 25 min/comida
- **Compatibilidad:** Liconsa, Diconsa, Bienestar
- **Lista tiendita:** Frijoles, arroz, huevos, lentejas, nopales
- **Enfoque:** Proteína vegetal abundante, granos enteros

---

## 🛠️ Reproducibilidad

```bash
# Ejecutar pipeline completo
python run_pipeline.py

# O por fase
python src/ml/01_initial_regression.py
python src/ml/02_ml_with_inegi.py
python src/ml/03_meal_plan.py
```

### Tiempo de Ejecución
- Fase 1: ~2 segundos
- Fase 2: ~6 segundos
- Fase 3: ~3 segundos
- **Total: ~12 segundos**

### Outputs Generados
```
outputs/
├── 01_metrics.json
├── 01_linear_regression_results.png
├── 02_metrics.json
├── 02_ml_results.png
├── 02_feature_importance.csv
├── 03_meal_plans.json
├── 03_meal_plans.png
├── 03_cluster_profiles.csv
├── processed_survey_data.csv
└── integrated_dataset.csv
```

### Dashboard
🌐 **Visualización interactiva:** `src/web/results_dashboard.html`

---

## 📚 Próximos Pasos Recomendados

### Validación
1. **Validar con datos reales** - cuando estén disponibles los CSVs completos de DENUE/ENSANUT/ENIGH
2. **A/B testing** - probar planes alimenticios con familias piloto en SLP
3. **Field study** - medir adopción en tienditas locales

### Extensiones
4. **Modelo geoespacial** - usar lat/lon del DENUE para análisis a nivel municipio
5. **Integración Bass Diffusion** - conectar segmentos con modelo de adopción
6. **Recommender system** - sistema de recomendación de productos saludables por tiendita

### Producción
7. **API REST** - exponer modelo como servicio
8. **App móvil** - plan alimenticio personalizado on-the-go
9. **Partnership FEMSA/OXXO** - integrar con inventario de tienditas reales

---

## 🏆 Equipo

| Rol | Miembro | Aporte |
|-----|---------|--------|
| **Deep Research** | Fernanda Ita | Análisis crítico, marco teórico |
| **Data Collection** | Alexis Marcos | Encuestas, trabajo de campo |
| **Data Science** | Carlos Torres | ML, pipeline, full-stack |

**Institución:** Tec de Monterrey, Campus San Luis Potosí  
**Programa:** Ingeniería Mecatrónica  
**Competencia:** MIT LiftLab National 2025
