"""
Procesa los ZIPs del DENUE localmente y genera un CSV agregado por estado.
Ejecutar en la misma carpeta donde están los ZIPs.

Uso:
    python procesar_denue_local.py

Output: denue_resumen_por_estado.csv  (~2 KB)
        Súbelo al chat de Claude.
"""

import pandas as pd
import zipfile
import glob
import os

print("=" * 60)
print("PROCESAMIENTO LOCAL DENUE")
print("=" * 60)

# Detecta los ZIPs en el directorio actual
zips = sorted(glob.glob("denue_*.zip"))
if not zips:
    print("❌ No encontré ningún ZIP que empiece con 'denue_' en esta carpeta.")
    print("   Asegúrate de correr este script donde están los ZIPs.")
    exit(1)

print(f"\n✅ Encontré {len(zips)} ZIPs:")
for z in zips:
    print(f"   - {z}  ({os.path.getsize(z)/1024/1024:.1f} MB)")

# Carga todos los CSVs de los ZIPs
all_data = []
for zippath in zips:
    print(f"\n📂 Procesando {zippath}...")
    with zipfile.ZipFile(zippath) as z:
        for name in z.namelist():
            if name.lower().endswith('.csv'):
                print(f"   Leyendo {name}...")
                # DENUE usa encoding latin-1
                with z.open(name) as f:
                    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
                df['__source_zip'] = zippath
                df['__source_csv'] = name
                all_data.append(df)
                print(f"      {len(df):,} registros, {df.shape[1]} columnas")

if not all_data:
    print("❌ No se encontraron CSVs dentro de los ZIPs.")
    exit(1)

# Combinar todo
print(f"\n🔄 Combinando {len(all_data)} archivos...")
df_full = pd.concat(all_data, ignore_index=True)
print(f"   Total: {len(df_full):,} registros")

# Identificar columnas clave (los nombres pueden tener variaciones)
def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

def find_col_any(df, *keyword_groups):
    """Intenta múltiples combinaciones de keywords."""
    for keywords in keyword_groups:
        result = find_col(df, *keywords)
        if result:
            return result
    return None

print(f"\n🔍 Columnas disponibles en el CSV:")
for col in df_full.columns:
    print(f"   '{col}'")

col_entidad = find_col_any(df_full,
    ('entidad', 'federativa'),
    ('nombre', 'entidad'),
    ('entidad',),
    ('estado',))
col_municipio = find_col_any(df_full,
    ('nombre', 'municipio'),
    ('municipio',))
col_scian = find_col_any(df_full,
    ('clase', 'actividad', 'scian'),
    ('codigo', 'clase', 'actividad'),
    ('codigo', 'clase'),
    ('codigo', 'scian'),
    ('scian',),
    ('clase', 'actividad'),
    ('clase',))
col_nombre_clase = find_col_any(df_full,
    ('nombre', 'clase', 'actividad'),
    ('nombre', 'clase'),
    ('descripcion', 'clase'))
col_personal = find_col_any(df_full,
    ('estrato', 'personal'),
    ('descripcion', 'estrato'),
    ('personal',),
    ('estrato',))
col_lat = find_col(df_full, 'latitud')
col_lon = find_col(df_full, 'longitud')

print(f"\n📊 Columnas identificadas:")
print(f"   Entidad:    {col_entidad}")
print(f"   Municipio:  {col_municipio}")
print(f"   SCIAN:      {col_scian}")
print(f"   Clase:      {col_nombre_clase}")
print(f"   Personal:   {col_personal}")

if not col_entidad:
    print("\n❌ No se pudo detectar la columna de Entidad/Estado.")
    print("   Revisa los nombres de columnas arriba y ajusta el script.")
    exit(1)

# Agregar por estado y por código SCIAN
print(f"\n📈 Agregando por estado...")

# Limpiar entidad (a veces tiene espacios al final)
df_full[col_entidad] = df_full[col_entidad].astype(str).str.strip()

# Resumen por estado
if col_scian:
    agg = df_full.groupby([col_entidad, col_scian]).size().reset_index(name='total_tiendas')
    pivot = agg.pivot_table(
        index=col_entidad,
        columns=col_scian,
        values='total_tiendas',
        fill_value=0
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={col_entidad: 'Entidad'})
    scian_cols = [c for c in pivot.columns if c != 'Entidad']
    pivot['Total_Tiendas'] = pivot[scian_cols].sum(axis=1)
else:
    print("   ⚠️  Columna SCIAN no detectada — contando solo totales por estado.")
    total = df_full.groupby(col_entidad).size().reset_index(name='Total_Tiendas')
    pivot = total.rename(columns={col_entidad: 'Entidad'})

# Top municipios por entidad (top 5) - para tener idea de distribución
if col_municipio:
    top_municipios = (
        df_full.groupby([col_entidad, col_municipio]).size()
        .reset_index(name='count')
        .sort_values(['count'], ascending=False)
        .groupby(col_entidad).head(3)
        .groupby(col_entidad)
        .apply(lambda x: ', '.join(x[col_municipio].astype(str).head(3)))
        .reset_index(name='Top3_Municipios')
    )
    top_municipios = top_municipios.rename(columns={col_entidad: 'Entidad'})
    pivot = pivot.merge(top_municipios, on='Entidad', how='left')

# Personal ocupado (si hay)
if col_personal:
    personal_summary = (
        df_full.groupby(col_entidad)[col_personal]
        .agg(lambda x: x.value_counts().to_dict())
        .reset_index()
    )
    personal_summary = personal_summary.rename(columns={col_entidad: 'Entidad', col_personal: 'Personal_Dist'})
    # Quédate solo con un proxy: % tiendas pequeñas (0-5 personas)
    def pct_small(d):
        if not isinstance(d, dict): return 0
        total = sum(d.values())
        small = sum(v for k, v in d.items() if isinstance(k, str) and ('0 a 5' in k or '6 a 10' in k))
        return round(small/total*100, 1) if total else 0
    personal_summary['Pct_Tiendas_Pequenas'] = personal_summary['Personal_Dist'].apply(pct_small)
    personal_summary = personal_summary[['Entidad', 'Pct_Tiendas_Pequenas']]
    pivot = pivot.merge(personal_summary, on='Entidad', how='left')

# Guardar
output = "denue_resumen_por_estado.csv"
pivot.to_csv(output, index=False, encoding='utf-8')

print(f"\n✅ LISTO!")
print(f"   Archivo: {output}")
print(f"   Tamaño:  {os.path.getsize(output)/1024:.1f} KB")
print(f"   Estados: {len(pivot)}")
print(f"\n   Total establecimientos: {pivot['Total_Tiendas'].sum():,}")
print(f"\n📤 Sube '{output}' al chat de Claude y listo.")
print(f"\nPreview:")
print(pivot.head(10).to_string(index=False))
