"""
02 - Analisis exploratorio ponderado de la base analitica de anemia infantil.

Entrada : Base_Analitica/base_anemia_endes2025.csv
Salidas : Resultados/tablas/*.csv  y  Resultados/figuras/*.png

Todas las prevalencias usan el factor de expansion de la ENDES. Las frecuencias
(n) se reportan sin ponderar porque describen la muestra, no la poblacion.
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.environ.get("PROY_DIR", ".")
IN_CSV = os.path.join(BASE, "Base_Analitica", "base_anemia_endes2025.csv")
OUT = os.path.join(BASE, "Resultados")
T = os.path.join(OUT, "tablas")
os.makedirs(T, exist_ok=True)

df = pd.read_csv(IN_CSV, encoding="utf-8-sig", low_memory=False)
df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

log = []
def say(m=""):
    print(m)
    log.append(str(m))


def prev(d, col="anemia", w="factor"):
    """Prevalencia ponderada en porcentaje."""
    d = d[d[col].notna()]
    if len(d) == 0:
        return np.nan
    return float(np.average(d[col], weights=d[w])) * 100


def ic95(d, col="anemia", w="factor"):
    """Intervalo de confianza aproximado considerando el efecto de diseno."""
    d = d[d[col].notna()]
    p = np.average(d[col], weights=d[w])
    n = len(d)
    # efecto de diseno aproximado por la variabilidad de los pesos (Kish)
    wn = d[w] / d[w].mean()
    deff = 1 + wn.var()
    n_eff = n / deff
    se = np.sqrt(p * (1 - p) / n_eff)
    return (max(0, (p - 1.96 * se)) * 100, min(1, (p + 1.96 * se)) * 100)


def tabla(d, var, nombre_archivo, orden=None):
    filas = []
    for k, g in d.groupby(var, observed=True):
        lo, hi = ic95(g)
        filas.append({
            var: k, "n": len(g),
            "prevalencia_anemia": round(prev(g), 1),
            "ic95_inf": round(lo, 1), "ic95_sup": round(hi, 1),
            "prevalencia_oms2024": round(prev(g, "anemia_oms"), 1),
        })
    t = pd.DataFrame(filas)
    if orden:
        t[var] = pd.Categorical(t[var], categories=orden, ordered=True)
        t = t.sort_values(var)
    else:
        t = t.sort_values("prevalencia_anemia", ascending=False)
    t.to_csv(os.path.join(T, nombre_archivo), index=False, encoding="utf-8-sig")
    return t


say("=" * 74)
say("ANALISIS EXPLORATORIO - ANEMIA INFANTIL 6 A 59 MESES - ENDES 2025")
say("=" * 74)
say(f"\nNinos en la base: {len(df):,}")
lo, hi = ic95(df)
say(f"Prevalencia nacional (RM 363-2022-MINSA): {prev(df):.1f} %  (IC95 {lo:.1f} - {hi:.1f})")
say(f"Prevalencia nacional (directriz OMS 2024): {prev(df,'anemia_oms'):.1f} %")
d35 = df[df.edad_meses.between(6, 35)]
say(f"Prevalencia 6-35 meses (indicador mas difundido): {prev(d35):.1f} %")

say("\n--- SEVERIDAD ---")
sev = (df.assign(peso=df.factor).groupby("nivel_anemia")["peso"].sum() / df.factor.sum() * 100)
for k in ["Grave", "Moderada", "Leve", "Sin anemia"]:
    if k in sev.index:
        say(f"   {k:12s} {sev[k]:5.1f} %")
sev.rename("porcentaje").round(2).to_csv(os.path.join(T, "severidad.csv"), encoding="utf-8-sig")

say("\n--- POR DEPARTAMENTO (ordenado) ---")
t_dep = tabla(df, "departamento", "prevalencia_departamento.csv")
for _, r in t_dep.iterrows():
    say(f"   {r['departamento']:16s} {r['prevalencia_anemia']:5.1f} %   (n={int(r['n']):,})")

say("\n--- POR ALTITUD ---")
orden_alt = ["Costa baja (<1000 m)", "Intermedia (1000-2499 m)",
             "Alta (2500-3499 m)", "Muy alta (3500+ m)"]
t_alt = tabla(df, "altitud_cat", "prevalencia_altitud.csv", orden_alt)
for _, r in t_alt.iterrows():
    say(f"   {str(r['altitud_cat']):28s} {r['prevalencia_anemia']:5.1f} %   (n={int(r['n']):,})")

say("\n--- POR QUINTIL DE RIQUEZA ---")
orden_q = ["Mas pobre", "Pobre", "Medio", "Rico", "Mas rico"]
t_q = tabla(df, "quintil_riqueza_lbl", "prevalencia_quintil.csv", orden_q)
for _, r in t_q.iterrows():
    say(f"   {str(r['quintil_riqueza_lbl']):12s} {r['prevalencia_anemia']:5.1f} %   (n={int(r['n']):,})")

say("\n--- POR EDAD ---")
orden_e = ["6-11 m", "12-23 m", "24-35 m", "36-47 m", "48-59 m"]
t_e = tabla(df, "grupo_edad", "prevalencia_edad.csv", orden_e)
for _, r in t_e.iterrows():
    say(f"   {str(r['grupo_edad']):10s} {r['prevalencia_anemia']:5.1f} %   (n={int(r['n']):,})")

say("\n--- OTROS CORTES ---")
for var, arch in [("area", "prevalencia_area.csv"),
                  ("region_natural", "prevalencia_region.csv"),
                  ("educacion_madre", "prevalencia_educacion_madre.csv"),
                  ("sexo", "prevalencia_sexo.csv")]:
    t = tabla(df, var, arch)
    say(f"\n   {var}:")
    for _, r in t.iterrows():
        say(f"     {str(r[var]):22s} {r['prevalencia_anemia']:5.1f} %   (n={int(r['n']):,})")

say("\n--- ANEMIA MATERNA vs ANEMIA INFANTIL ---")
dm = df[df.anemia_madre.notna()]
for k, g in dm.groupby("anemia_madre"):
    etq = "Madre con anemia" if k == 1 else "Madre sin anemia"
    say(f"   {etq:20s} {prev(g):5.1f} %   (n={len(g):,})")
razon = prev(dm[dm.anemia_madre == 1]) / prev(dm[dm.anemia_madre == 0])
say(f"   Razon de prevalencias: {razon:.2f}")
(dm.groupby("anemia_madre")
   .apply(lambda g: pd.Series({"n": len(g), "prevalencia_hijo": round(prev(g), 1)}),
          include_groups=False)
   .to_csv(os.path.join(T, "anemia_materna.csv"), encoding="utf-8-sig"))

say("\n--- SANEAMIENTO ---")
filas = []
for var, etq in [("agua_red_publica", "Agua de red publica"),
                 ("desague_red", "Desague de red publica"),
                 ("piso_tierra", "Piso de tierra"),
                 ("combustible_contaminante", "Cocina con combustible contaminante"),
                 ("electricidad", "Tiene electricidad")]:
    g = df[df[var].notna()]
    p1, p0 = prev(g[g[var] == 1]), prev(g[g[var] == 0])
    say(f"   {etq:38s} si: {p1:5.1f} %   no: {p0:5.1f} %   brecha: {p0-p1:+5.1f} pp")
    filas.append({"condicion": etq, "prev_si": round(p1, 1), "prev_no": round(p0, 1),
                  "brecha_pp": round(p0 - p1, 1)})
pd.DataFrame(filas).to_csv(os.path.join(T, "saneamiento.csv"), index=False, encoding="utf-8-sig")

say("\n--- BRECHA EXTREMA ---")
peor = t_dep.iloc[0]
mejor = t_dep.iloc[-1]
say(f"   {peor['departamento']} ({peor['prevalencia_anemia']:.1f} %) vs "
    f"{mejor['departamento']} ({mejor['prevalencia_anemia']:.1f} %) = "
    f"{peor['prevalencia_anemia'] - mejor['prevalencia_anemia']:.1f} puntos porcentuales")

resumen = {
    "n_ninos": int(len(df)),
    "prev_nacional": round(prev(df), 1),
    "prev_nacional_ic": [round(lo, 1), round(hi, 1)],
    "prev_oms2024": round(prev(df, "anemia_oms"), 1),
    "prev_6a35": round(prev(d35), 1),
    "prev_6a35_oms": round(prev(d35, "anemia_oms"), 1),
    "severidad": {k: round(float(v), 1) for k, v in sev.items()},
    "razon_anemia_materna": round(razon, 2),
}
with open(os.path.join(OUT, "resumen_exploratorio.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT, "reporte_exploratorio.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))

say(f"\nTablas guardadas en {T}")
