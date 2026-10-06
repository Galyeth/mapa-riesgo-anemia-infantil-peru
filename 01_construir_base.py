"""
Construccion de la base analitica para el modelo de riesgo de anemia infantil.

Fuente unica: ENDES 2025 - Instituto Nacional de Estadistica e Informatica (INEI)
Modulos utilizados: 1638 (RECH6, RECH5), 1629 (RECH0), 1630 (RECH23)

Unidad de analisis : un niño de 6 a 59 meses con medicion valida de hemoglobina
Salida             : base_anemia_endes2025.csv / .db / diccionario / reporte

No se modifica ningun archivo original. Solo lee.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. Rutas
# --------------------------------------------------------------------------
BASE_IN = os.environ.get("ENDES_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "./Base_Analitica")
os.makedirs(OUT_DIR, exist_ok=True)

F_RECH6 = os.path.join(BASE_IN, "1038-Modulo1638", "RECH6_2025.csv")
F_RECH5 = os.path.join(BASE_IN, "1038-Modulo1638", "RECH5_2025.csv")
F_RECH0 = os.path.join(BASE_IN, "1038-Modulo1629", "RECH0_2025.csv")
F_RECH23 = os.path.join(BASE_IN, "1038-Modulo1630", "RECH23_2025.csv")

log = []
def say(msg=""):
    print(msg)
    log.append(str(msg))


def num(s):
    """Convierte a numerico tolerando blancos y textos."""
    return pd.to_numeric(s, errors="coerce")


def na_if(serie, codigos):
    """Reemplaza codigos de no-respuesta por NaN."""
    return serie.mask(serie.isin(codigos))


# --------------------------------------------------------------------------
# 1. Lectura
# --------------------------------------------------------------------------
say("=" * 72)
say("BASE ANALITICA - ANEMIA INFANTIL - ENDES 2025")
say("=" * 72)

rech6 = pd.read_csv(F_RECH6, encoding="utf-8-sig", low_memory=False)
rech5 = pd.read_csv(F_RECH5, encoding="utf-8-sig", low_memory=False)
rech0 = pd.read_csv(F_RECH0, encoding="utf-8-sig", low_memory=False)
rech23 = pd.read_csv(F_RECH23, encoding="utf-8-sig", low_memory=False)

say("\n1. ARCHIVOS LEIDOS (originales, sin modificar)")
for nombre, d in [("RECH6 ninos", rech6), ("RECH5 madres", rech5),
                  ("RECH0 hogar", rech0), ("RECH23 vivienda", rech23)]:
    say(f"   {nombre:18s} {d.shape[0]:>7,} filas x {d.shape[1]:>3} columnas")

# --------------------------------------------------------------------------
# 2. Diccionarios de etiquetas (tomados del diccionario oficial del INEI)
# --------------------------------------------------------------------------
DEPARTAMENTOS = {
    1: "Amazonas", 2: "Ancash", 3: "Apurimac", 4: "Arequipa", 5: "Ayacucho",
    6: "Cajamarca", 7: "Callao", 8: "Cusco", 9: "Huancavelica", 10: "Huanuco",
    11: "Ica", 12: "Junin", 13: "La Libertad", 14: "Lambayeque", 15: "Lima",
    16: "Loreto", 17: "Madre de Dios", 18: "Moquegua", 19: "Pasco", 20: "Piura",
    21: "Puno", 22: "San Martin", 23: "Tacna", 24: "Tumbes", 25: "Ucayali",
}
REGION_NATURAL = {1: "Lima Metropolitana", 2: "Resto Costa", 3: "Sierra", 4: "Selva"}
AREA = {1: "Urbano", 2: "Rural"}
NIVEL_ANEMIA = {1: "Grave", 2: "Moderada", 3: "Leve", 4: "Sin anemia"}
EDUC_MADRE = {0: "Sin educacion", 1: "Primaria", 2: "Secundaria", 3: "Superior"}
QUINTIL = {1: "Mas pobre", 2: "Pobre", 3: "Medio", 4: "Rico", 5: "Mas rico"}
FUENTE_AGUA = {
    11: "Red dentro de la vivienda", 12: "Red fuera de la vivienda",
    13: "Pilon o grifo publico", 21: "Pozo en la vivienda", 22: "Pozo publico",
    41: "Manantial (puquio)", 43: "Rio, acequia o laguna", 51: "Agua de lluvia",
    61: "Camion cisterna", 71: "Agua embotellada", 96: "Otro",
}
SERVICIO_HIG = {
    11: "Red publica dentro de la vivienda", 12: "Red publica fuera de la vivienda",
    21: "Letrina mejorada ventilada", 22: "Pozo septico",
    23: "Letrina (pozo ciego o negro)", 24: "Letrina mejorada colgada",
    31: "Rio, acequia o canal", 32: "Sin servicio", 96: "Otro",
}
PISO = {
    11: "Tierra o arena", 21: "Madera", 31: "Parquet o madera pulida",
    32: "Laminas asfalticas o vinilicos", 33: "Losetas o terrazos",
    34: "Cemento o ladrillo", 96: "Otro",
}
COMBUSTIBLE = {
    1: "Electricidad", 2: "GLP", 3: "Gas natural", 5: "Kerosene",
    6: "Carbon vegetal", 7: "Carbon mineral", 8: "Lena",
    9: "Cana o arbustos secos", 10: "Residuos agricolas", 11: "Bosta",
    95: "No cocina", 96: "Otro",
}

# --------------------------------------------------------------------------
# 3. Tabla base: ninos (RECH6)
# --------------------------------------------------------------------------
n = pd.DataFrame()
n["hhid"] = rech6["HHID"]
n["orden_nino"] = num(rech6["HC0"])
n["anio"] = num(rech6["ID1"])

n["edad_meses"] = num(rech6["HC1"])
n["sexo"] = num(rech6["HC27"]).map({1: "Hombre", 2: "Mujer"})

n["peso_kg"] = na_if(num(rech6["HC2"]), [9999, 9998, 9996]) / 10
n["talla_cm"] = na_if(num(rech6["HC3"]), [9999, 9998, 9996]) / 10
n["z_talla_edad"] = na_if(num(rech6["HC70"]), [9999, 9998, 9996]) / 100
n["z_peso_edad"] = na_if(num(rech6["HC71"]), [9999, 9998, 9996]) / 100
n["z_peso_talla"] = na_if(num(rech6["HC72"]), [9999, 9998, 9996]) / 100

n["orden_nacimiento"] = na_if(num(rech6["HC64"]), [99, 98])
n["intervalo_nacimiento_meses"] = na_if(num(rech6["HC63"]), [999, 998])

# hemoglobina --------------------------------------------------------------
n["resultado_medicion"] = num(rech6["HC55"])
n["hb_cruda"] = na_if(num(rech6["HC53"]), [999, 998, 996]) / 10
n["hb_ajustada"] = na_if(num(rech6["HC56"]), [999, 998, 996]) / 10
n["hb_ajustada_oms"] = na_if(num(rech6["HC56A"]), [999, 998, 996]) / 10

nivel = na_if(num(rech6["HC57"]), [9, 8])
nivel_oms = na_if(num(rech6["HC57A"]), [9, 8])
n["nivel_anemia"] = nivel.map(NIVEL_ANEMIA)
n["nivel_anemia_oms"] = nivel_oms.map(NIVEL_ANEMIA)
n["anemia"] = nivel.isin([1, 2, 3]).astype("float").where(nivel.notna())
n["anemia_oms"] = nivel_oms.isin([1, 2, 3]).astype("float").where(nivel_oms.notna())

# madre (lo que ya viene en RECH6) ------------------------------------------
educ = na_if(num(rech6["HC61"]), [8, 9])
n["educacion_madre"] = educ.map(EDUC_MADRE)
n["educacion_madre_num"] = educ
n["_orden_madre"] = na_if(num(rech6["HC60"]), [993, 994, 995])

# --------------------------------------------------------------------------
# 4. Hogar (RECH0): altitud, geografia, diseno muestral
# --------------------------------------------------------------------------
h = pd.DataFrame()
h["hhid"] = rech0["HHID"]
h["ubigeo"] = rech0["UBIGEO"].astype(str).str.strip().str.zfill(6)
h["departamento"] = num(rech0["HV024"]).map(DEPARTAMENTOS)
h["departamento_cod"] = num(rech0["HV024"])
h["area"] = num(rech0["HV025"]).map(AREA)
h["altitud_m"] = na_if(num(rech0["HV040"]), [9999, 9998])
h["latitud"] = num(rech0["LATITUDY"])
h["longitud"] = num(rech0["LONGITUDX"])
h["tam_hogar"] = num(rech0["HV009"])
h["n_menores5"] = num(rech0["HV014"])
h["factor"] = num(rech0["HV005"]) / 1_000_000
h["conglomerado"] = num(rech0["HV021"])
h["estrato"] = num(rech0["HV022"])

# --------------------------------------------------------------------------
# 5. Vivienda (RECH23): riqueza, agua, saneamiento
# --------------------------------------------------------------------------
v = pd.DataFrame()
v["hhid"] = rech23["HHID"]
v["region_natural"] = num(rech23["SHREGION"]).map(REGION_NATURAL)

q = na_if(num(rech23["HV270"]), [9])
v["quintil_riqueza"] = q
v["quintil_riqueza_lbl"] = q.map(QUINTIL)
v["indice_riqueza"] = num(rech23["HV271"])

agua = na_if(num(rech23["HV201"]), [99, 98])
v["fuente_agua"] = agua.map(FUENTE_AGUA)
v["agua_red_publica"] = agua.isin([11, 12, 13]).astype("float").where(agua.notna())
v["agua_dentro_vivienda"] = agua.eq(11).astype("float").where(agua.notna())

t_agua = num(rech23["HV204"])
v["tiempo_agua_min"] = t_agua.replace({996: 0}).mask(t_agua.isin([998, 999]))

sh = na_if(num(rech23["HV205"]), [99, 98])
v["servicio_higienico"] = sh.map(SERVICIO_HIG)
v["desague_red"] = sh.isin([11, 12]).astype("float").where(sh.notna())
v["sin_servicio_higienico"] = sh.isin([31, 32]).astype("float").where(sh.notna())

v["electricidad"] = na_if(num(rech23["HV206"]), [9])

piso = na_if(num(rech23["HV213"]), [99, 98])
v["piso"] = piso.map(PISO)
v["piso_tierra"] = piso.eq(11).astype("float").where(piso.notna())

comb = na_if(num(rech23["HV226"]), [99, 98])
v["combustible_cocina"] = comb.map(COMBUSTIBLE)
v["combustible_contaminante"] = comb.isin([5, 6, 7, 8, 9, 10, 11]).astype("float").where(comb.notna())

v["sexo_jefe_hogar"] = na_if(num(rech23["HV219"]), [9]).map({1: "Hombre", 2: "Mujer"})
v["edad_jefe_hogar"] = na_if(num(rech23["HV220"]), [97, 98, 99])

# --------------------------------------------------------------------------
# 6. Madre (RECH5): anemia materna
# --------------------------------------------------------------------------
m = pd.DataFrame()
m["hhid"] = rech5["HHID"]
m["_orden_madre"] = num(rech5["HA0"])
m["edad_madre"] = na_if(num(rech5["HA1"]), [99, 98])
m["imc_madre"] = na_if(num(rech5["HA40"]), [9999, 9998, 9996]) / 100
m["hb_madre"] = na_if(num(rech5["HA56"]), [999, 998, 996]) / 10
nivel_m = na_if(num(rech5["HA57"]), [9, 8])
m["nivel_anemia_madre"] = nivel_m.map(NIVEL_ANEMIA)
m["anemia_madre"] = nivel_m.isin([1, 2, 3]).astype("float").where(nivel_m.notna())

# --------------------------------------------------------------------------
# 7. Uniones (todas por HHID: misma encuesta, misma llave)
# --------------------------------------------------------------------------
say("\n2. UNIONES (verificadas fila a fila)")
antes = len(n)
df = n.merge(h, on="hhid", how="left", validate="m:1")
say(f"   + RECH0  : {antes:,} -> {len(df):,} filas | sin pareja: {df['altitud_m'].isna().sum():,}")

df = df.merge(v, on="hhid", how="left", validate="m:1")
say(f"   + RECH23 : {antes:,} -> {len(df):,} filas | sin pareja: {df['quintil_riqueza'].isna().sum():,}")

df = df.merge(m, on=["hhid", "_orden_madre"], how="left", validate="m:1")
say(f"   + RECH5  : {antes:,} -> {len(df):,} filas | madre no enlazada: {df['edad_madre'].isna().sum():,}")

# --------------------------------------------------------------------------
# 8. Universo del indicador
# --------------------------------------------------------------------------
say("\n3. FILTRO AL UNIVERSO DEL INDICADOR")
say(f"   Ninos 0-59 meses en RECH6            : {len(df):,}")

df = df[df["edad_meses"].between(6, 59)].copy()
say(f"   Ninos 6-59 meses                     : {len(df):,}")

sin_medicion = df["anemia"].isna().sum()
df = df[df["anemia"].notna()].copy()
say(f"   Excluidos por no tener hemoglobina   : {sin_medicion:,}")
say(f"   BASE FINAL (universo de modelado)    : {len(df):,}")

# --------------------------------------------------------------------------
# 9. Variables derivadas
# --------------------------------------------------------------------------
df["id_nino"] = df["hhid"].astype(str) + "-" + df["orden_nino"].astype(int).astype(str)

df["grupo_edad"] = pd.cut(
    df["edad_meses"], bins=[5, 11, 23, 35, 47, 59],
    labels=["6-11 m", "12-23 m", "24-35 m", "36-47 m", "48-59 m"],
)
df["altitud_cat"] = pd.cut(
    df["altitud_m"], bins=[-100, 999, 2499, 3499, 6000],
    labels=["Costa baja (<1000 m)", "Intermedia (1000-2499 m)",
            "Alta (2500-3499 m)", "Muy alta (3500+ m)"],
)
df["desnutricion_cronica"] = (df["z_talla_edad"] < -2).astype("float").where(df["z_talla_edad"].notna())
df["anemia"] = df["anemia"].astype("int8")

COLS = [
    "id_nino", "hhid", "orden_nino", "anio",
    "anemia", "anemia_oms", "nivel_anemia", "nivel_anemia_oms",
    "hb_ajustada", "hb_ajustada_oms", "hb_cruda",
    "edad_meses", "grupo_edad", "sexo", "peso_kg", "talla_cm",
    "z_talla_edad", "z_peso_edad", "z_peso_talla", "desnutricion_cronica",
    "orden_nacimiento", "intervalo_nacimiento_meses",
    "educacion_madre", "educacion_madre_num", "edad_madre", "imc_madre",
    "anemia_madre", "nivel_anemia_madre", "hb_madre",
    "ubigeo", "departamento", "departamento_cod", "region_natural", "area",
    "altitud_m", "altitud_cat", "latitud", "longitud",
    "quintil_riqueza", "quintil_riqueza_lbl", "indice_riqueza",
    "fuente_agua", "agua_red_publica", "agua_dentro_vivienda", "tiempo_agua_min",
    "servicio_higienico", "desague_red", "sin_servicio_higienico",
    "electricidad", "piso", "piso_tierra",
    "combustible_cocina", "combustible_contaminante",
    "tam_hogar", "n_menores5", "sexo_jefe_hogar", "edad_jefe_hogar",
    "factor", "conglomerado", "estrato",
]
df = df[COLS].reset_index(drop=True)

# --------------------------------------------------------------------------
# 10. Validacion: reproducir la prevalencia oficial
# --------------------------------------------------------------------------
def prev(d, col="anemia", w="factor"):
    d = d[d[col].notna()]
    return float(np.average(d[col], weights=d[w])) * 100

say("\n4. VALIDACION - PREVALENCIA DE ANEMIA (ponderada por factor de expansion)")
say(f"   6-59 meses  RM 363-2022-MINSA   : {prev(df):5.1f} %   (sin ponderar {df['anemia'].mean()*100:5.1f} %)")
say(f"   6-59 meses  Directriz OMS 2024  : {prev(df, 'anemia_oms'):5.1f} %")
d35 = df[df["edad_meses"].between(6, 35)]
say(f"   6-35 meses  RM 363-2022-MINSA   : {prev(d35):5.1f} %   <- comparar con el dato publicado por el INEI")
say(f"   6-35 meses  Directriz OMS 2024  : {prev(d35, 'anemia_oms'):5.1f} %")

say("\n   Prevalencia por area:")
for a, g in df.groupby("area"):
    say(f"     {a:8s} {prev(g):5.1f} %   (n = {len(g):,})")
say("\n   Prevalencia por categoria de altitud:")
for a, g in df.groupby("altitud_cat", observed=True):
    say(f"     {str(a):26s} {prev(g):5.1f} %   (n = {len(g):,})")
say("\n   Top 5 departamentos con mayor prevalencia:")
dep = (df.groupby("departamento")
         .apply(lambda g: pd.Series({"prev": prev(g), "n": len(g)}), include_groups=False)
         .sort_values("prev", ascending=False))
for d_, r in dep.head(5).iterrows():
    say(f"     {d_:16s} {r['prev']:5.1f} %   (n = {int(r['n']):,})")

say("\n5. COMPLETITUD DE LA BASE FINAL")
faltantes = df.isna().sum()
faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
if len(faltantes) == 0:
    say("   Sin valores faltantes.")
for c, k in faltantes.items():
    say(f"   {c:28s} {k:>6,}  ({k/len(df)*100:4.1f} %)")

# --------------------------------------------------------------------------
# 11. Salidas
# --------------------------------------------------------------------------
csv_out = os.path.join(OUT_DIR, "base_anemia_endes2025.csv")
db_out = os.path.join(OUT_DIR, "base_anemia_endes2025.db")

df.to_csv(csv_out, index=False, encoding="utf-8-sig")

con = sqlite3.connect(db_out)
df.assign(grupo_edad=df["grupo_edad"].astype(str),
          altitud_cat=df["altitud_cat"].astype(str)).to_sql(
    "anemia", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS ix_dep ON anemia(departamento)")
con.execute("CREATE INDEX IF NOT EXISTS ix_ubi ON anemia(ubigeo)")
con.commit()
con.close()

DESCRIPCIONES = {
    "id_nino": "Identificador unico del nino (hogar + numero de orden)",
    "hhid": "Identificador del hogar - llave de union entre archivos ENDES",
    "orden_nino": "Numero de orden del nino dentro del hogar",
    "anio": "Anio de la encuesta",
    "anemia": "VARIABLE OBJETIVO. 1 = con anemia, 0 = sin anemia (RM 363-2022-MINSA)",
    "anemia_oms": "Variable objetivo alternativa segun la directriz OMS 2024 / RM 251-2024-MINSA",
    "nivel_anemia": "Severidad: Grave, Moderada, Leve, Sin anemia (RM 363-2022-MINSA)",
    "nivel_anemia_oms": "Severidad segun la directriz OMS 2024",
    "hb_ajustada": "Hemoglobina ajustada por altitud en g/dL (RM 363-2022-MINSA)",
    "hb_ajustada_oms": "Hemoglobina ajustada por altitud en g/dL (directriz OMS 2024)",
    "hb_cruda": "Hemoglobina medida sin ajustar por altitud en g/dL",
    "edad_meses": "Edad del nino en meses (universo: 6 a 59)",
    "grupo_edad": "Edad agrupada en cinco tramos",
    "sexo": "Sexo del nino",
    "peso_kg": "Peso en kilogramos",
    "talla_cm": "Talla en centimetros",
    "z_talla_edad": "Puntaje Z talla para la edad (patron OMS)",
    "z_peso_edad": "Puntaje Z peso para la edad (patron OMS)",
    "z_peso_talla": "Puntaje Z peso para la talla (patron OMS)",
    "desnutricion_cronica": "1 si el puntaje Z talla/edad es menor a -2",
    "orden_nacimiento": "Orden de nacimiento del nino",
    "intervalo_nacimiento_meses": "Meses transcurridos desde el nacimiento anterior",
    "educacion_madre": "Nivel educativo mas alto alcanzado por la madre",
    "educacion_madre_num": "Nivel educativo de la madre en codigo ordinal (0 a 3)",
    "edad_madre": "Edad de la madre en anios",
    "imc_madre": "Indice de masa corporal de la madre",
    "anemia_madre": "1 si la madre tiene anemia (RM 363-2022-MINSA)",
    "nivel_anemia_madre": "Severidad de la anemia materna",
    "hb_madre": "Hemoglobina ajustada por altitud de la madre en g/dL",
    "ubigeo": "Codigo de ubicacion geografica de 6 digitos, como texto",
    "departamento": "Departamento de residencia",
    "departamento_cod": "Codigo del departamento (1 a 25)",
    "region_natural": "Lima Metropolitana, Resto Costa, Sierra o Selva",
    "area": "Area de residencia urbana o rural",
    "altitud_m": "Altitud del conglomerado en metros sobre el nivel del mar",
    "altitud_cat": "Altitud agrupada en cuatro categorias",
    "latitud": "Latitud del conglomerado",
    "longitud": "Longitud del conglomerado",
    "quintil_riqueza": "Quintil del indice de riqueza del hogar (1 mas pobre a 5 mas rico)",
    "quintil_riqueza_lbl": "Quintil de riqueza con etiqueta",
    "indice_riqueza": "Puntaje continuo del indice de riqueza",
    "fuente_agua": "Fuente principal de agua para beber",
    "agua_red_publica": "1 si el agua proviene de red publica o grifo publico",
    "agua_dentro_vivienda": "1 si el agua llega por red dentro de la vivienda",
    "tiempo_agua_min": "Minutos para ir, recoger agua y volver (0 si esta en la vivienda)",
    "servicio_higienico": "Tipo de servicio higienico del hogar",
    "desague_red": "1 si el hogar esta conectado a red publica de desague",
    "sin_servicio_higienico": "1 si el hogar no tiene servicio higienico",
    "electricidad": "1 si el hogar tiene electricidad",
    "piso": "Material predominante del piso de la vivienda",
    "piso_tierra": "1 si el piso es de tierra o arena",
    "combustible_cocina": "Combustible usado con mas frecuencia para cocinar",
    "combustible_contaminante": "1 si cocina con lena, carbon, kerosene, bosta o residuos",
    "tam_hogar": "Total de personas en el hogar",
    "n_menores5": "Numero de ninos menores de 5 anios en el hogar",
    "sexo_jefe_hogar": "Sexo del jefe del hogar",
    "edad_jefe_hogar": "Edad del jefe del hogar",
    "factor": "Factor de expansion (HV005 / 1000000). Usar para estimar prevalencias",
    "conglomerado": "Unidad primaria de muestreo. Usar para validacion agrupada",
    "estrato": "Estrato muestral",
}
ORIGEN = {}
for c in COLS:
    if c in ("id_nino", "grupo_edad", "altitud_cat", "desnutricion_cronica"):
        ORIGEN[c] = "Derivada"
    elif c in ("edad_madre", "imc_madre", "anemia_madre", "nivel_anemia_madre", "hb_madre"):
        ORIGEN[c] = "RECH5 (modulo 1638)"
    elif c in ("ubigeo", "departamento", "departamento_cod", "area", "altitud_m", "latitud",
               "longitud", "tam_hogar", "n_menores5", "factor", "conglomerado", "estrato"):
        ORIGEN[c] = "RECH0 (modulo 1629)"
    elif c in ("region_natural", "quintil_riqueza", "quintil_riqueza_lbl", "indice_riqueza",
               "fuente_agua", "agua_red_publica", "agua_dentro_vivienda", "tiempo_agua_min",
               "servicio_higienico", "desague_red", "sin_servicio_higienico", "electricidad",
               "piso", "piso_tierra", "combustible_cocina", "combustible_contaminante",
               "sexo_jefe_hogar", "edad_jefe_hogar"):
        ORIGEN[c] = "RECH23 (modulo 1630)"
    else:
        ORIGEN[c] = "RECH6 (modulo 1638)"

dic = pd.DataFrame({
    "variable": COLS,
    "descripcion": [DESCRIPCIONES.get(c, "") for c in COLS],
    "origen": [ORIGEN[c] for c in COLS],
    "tipo": [str(df[c].dtype) for c in COLS],
    "faltantes": [int(df[c].isna().sum()) for c in COLS],
    "pct_faltantes": [round(df[c].isna().mean() * 100, 1) for c in COLS],
})
dic_out = os.path.join(OUT_DIR, "diccionario_base_anemia.csv")
dic.to_csv(dic_out, index=False, encoding="utf-8-sig")

say("\n6. ARCHIVOS GENERADOS")
say(f"   {os.path.basename(csv_out):32s} {len(df):,} filas x {len(COLS)} columnas")
say(f"   {os.path.basename(db_out):32s} tabla SQL 'anemia'")
say(f"   {os.path.basename(dic_out):32s} {len(dic)} variables documentadas")

rep_out = os.path.join(OUT_DIR, "reporte_validacion.txt")
with open(rep_out, "w", encoding="utf-8") as f:
    f.write("\n".join(log))
say(f"   {os.path.basename(rep_out):32s} este mismo reporte")
say("\nListo. Los archivos originales no fueron modificados.")
