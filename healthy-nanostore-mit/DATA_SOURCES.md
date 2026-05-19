# Fuentes de Datos - Healthy Nanostore MIT

## ✅ Datos Disponibles en el Repositorio

### Encuestas Primarias
1. **Encuesta Tec Campus** → `src/data/survey_responses.csv` (104 respuestas)
2. **Encuesta Google Forms** → `src/data/forms_survey_responses.csv` (101 respuestas)

---

## 📥 Bases de Datos del INEGI - Cómo Descargar

### 1. DENUE 2024 (Nanostores/Tienditas - 1.1M registros)
**Portal de Descarga Masiva:** https://www.inegi.org.mx/app/descarga/default.html

- Seleccionar: Geografía → DENUE → 2024
- Formatos: Shapefile o CSV
- Incluye: Localización, actividad económica, tamaño de establecimiento
- **Nota:** El archivo es grande (~600 MB comprimido), requiere descompresión

**Alternativa - Mapa Interactivo:** https://www.inegi.org.mx/app/mapa/denue/default.aspx
(Para exploración, no descarga masiva)

---

### 2. ENSANUT 2023 (Salud y Nutrición - Nacional)
**Portal Oficial:** https://ensanut.insp.mx/encuestas/ensanutsin2023/descargas.php

- Descarga: Datos completos de la encuesta continua
- Incluye: Prevalencia diabetes (18%), sobrepeso/obesidad (70%), consumo F&V
- Documentos analíticos: https://ensanut.insp.mx/encuestas/ensanutcontinua2023/documentos_analiticos.php
- **Muestra:** 11,720+ hogares, 13,378+ adultos

---

### 3. ENIGH 2022 (Ingresos y Gastos de Hogares)
**Portal Oficial - Nueva Serie:** https://www.inegi.org.mx/programas/enigh/nc/2022/

- Descarga: Datos de 105,525 hogares
- Incluye: Gastos en alimentación por decil de ingresos
- Materiales relacionados: https://www.inegi.org.mx/rnm/index.php/catalog/901/related-materials
- **Formato:** CSV, Excel, SAS

---

### 4. Índice de Marginación 2020 (CONAPO)
**Portal Oficial - datos.gob.mx:** https://www.datos.gob.mx/dataset/indices_marginacion

**Descargas directas:**
- Por Entidad Federativa (Excel): http://conapo.segob.gob.mx/work/models/CONAPO/Datos_Abiertos/Entidad_Federativa/IME_2020.xls
- Por Municipio y Localidad (CSV): https://datos.gob.mx/busca/dataset/indice-de-marginacion-carencias-poblacionales-por-localidad-municipio-y-entidad

- Incluye: Índices socioeconómicos por región
- **Base:** Censo de Población y Vivienda 2020

---

## 🛠️ Recomendación para Subir Archivos Pesados

Si el ZIP de INEGI es > 100 MB:

### Opción 1: Git LFS (si tienes configurado)
```bash
git lfs install
git lfs track "*.zip"
git add src/data/inegi_datasets/
```

### Opción 2: Extraer solo lo necesario
- DENUE: Filtrar por entidades donde hicieron encuestas (ej: SLP, QRO, GTO)
- Esto reduce tamaño del 600 MB a 50-100 MB

### Opción 3: Referencias a URLs
- Documentar URLs en JSON con metadata
- Descargar bajo demanda en scripts de análisis

---

## 📊 Estado Actual del Proyecto

| Dataset | Filas | Formato | Estado |
|---------|-------|---------|--------|
| Survey Tec | 104 | CSV | ✅ En repo |
| Survey Forms | 101 | CSV | ✅ En repo |
| DENUE 2024 | 1,100,824 | Shapefile/CSV | 📥 Disponible (requiere descarga manual) |
| ENSANUT 2023 | 13,378+ | CSV/Excel | 📥 Disponible |
| ENIGH 2022 | 105,525 | CSV/Excel | 📥 Disponible |
| CONAPO 2020 | Variables | Excel/CSV | 📥 Disponible |

---

## 🎯 Próximos Pasos

1. **Análisis Inicial** (ahora mismo con encuestas que tenemos)
   - Descriptivos: edad, ocupación, hábitos
   - Insights: compra comida chatarra vs. tiempo disponible
   
2. **Descargar INEGI** (en paralelo)
   - Prioridad: DENUE (para distribución nanostores) + ENSANUT (calibración modelo)
   
3. **Integrar Datos**
   - Merge encuestas con INEGI por entidad
   - Regresión lineal inicial
   - Pipeline ML con features del INEGI

4. **Plan Alimenticio**
   - Con insights de Forms + análisis
   - Segmentado por tipo de familia (ingresos, estructura familiar)

