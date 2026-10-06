"""
04 - Mapa de riesgo y priorizacion territorial.

Entrada : Resultados/modelo/predicciones.csv
Salidas : Resultados/mapa/*.csv  (listos para Power BI)

Dos preguntas distintas, dos respuestas distintas:
  - Donde es mas probable que un niño tenga anemia  -> prevalencia / riesgo medio
  - Donde estan la mayoria de los niños con anemia  -> concentracion de casos
Un departamento pequeño y muy afectado y uno grande y medianamente afectado
piden intervenciones diferentes. El cuadrante de priorizacion los separa.
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.environ.get("PROY_DIR", ".")
PRED = os.path.join(BASE, "Resultados", "modelo", "predicciones.csv")
BASE_CSV = os.path.join(BASE, "Base_Analitica", "base_anemia_endes2025.csv")
OUT = os.path.join(BASE, "Resultados", "mapa")
os.makedirs(OUT, exist_ok=True)

log = []
def say(m=""):
    print(m)
    log.append(str(m))

p = pd.read_csv(PRED, encoding="utf-8-sig", low_memory=False)
b = pd.read_csv(BASE_CSV, encoding="utf-8-sig", low_memory=False)
p["ubigeo"] = p["ubigeo"].astype(str).str.zfill(6)
p = p.merge(b[["id_nino", "anemia_oms", "latitud", "longitud"]], on="id_nino", how="left")

W = p["factor"]
casos_tot = float((p["anemia"] * W).sum())

say("=" * 74)
say("MAPA DE RIESGO Y PRIORIZACION TERRITORIAL")
say("=" * 74)


def agregar(d, claves):
    g = d.groupby(claves, observed=True)
    out = g.apply(lambda x: pd.Series({
        "n_muestra": len(x),
        "prevalencia_obs": np.average(x.anemia, weights=x.factor) * 100,
        "riesgo_predicho": np.average(x.riesgo_predicho, weights=x.factor) * 100,
        "prevalencia_oms2024": np.average(x.anemia_oms.fillna(x.anemia), weights=x.factor) * 100,
        "altitud_media": np.average(x.altitud_m.fillna(x.altitud_m.median()), weights=x.factor),
        "casos_estimados": float((x.anemia * x.factor).sum()),
        "ninos_estimados": float(x.factor.sum()),
    }), include_groups=False).reset_index()
    out["pct_casos_pais"] = out.casos_estimados / casos_tot * 100
    for c in ["prevalencia_obs", "riesgo_predicho", "prevalencia_oms2024",
              "altitud_media", "pct_casos_pais"]:
        out[c] = out[c].round(1)
    return out


# --------------------------------------------------------------------------
# 1. Departamento
# --------------------------------------------------------------------------
dep = agregar(p, "departamento").sort_values("prevalencia_obs", ascending=False)

# cuadrante de priorizacion
med_prev = dep.prevalencia_obs.median()
med_carga = dep.pct_casos_pais.median()


def cuadrante(r):
    alta_p = r.prevalencia_obs >= med_prev
    alta_c = r.pct_casos_pais >= med_carga
    if alta_p and alta_c:
        return "1. Critico - alta prevalencia y alto volumen"
    if alta_p and not alta_c:
        return "2. Focalizado - alta prevalencia, pocos ninos"
    if not alta_p and alta_c:
        return "3. Volumen - prevalencia media, muchos ninos"
    return "4. Vigilancia"


dep["cuadrante"] = dep.apply(cuadrante, axis=1)
dep["brecha_definiciones_pp"] = (dep.prevalencia_obs - dep.prevalencia_oms2024).round(1)
dep["ranking"] = range(1, len(dep) + 1)
dep.to_csv(os.path.join(OUT, "riesgo_departamento.csv"), index=False, encoding="utf-8-sig")

say(f"\nMediana de prevalencia departamental: {med_prev:.1f} %")
say(f"Mediana de concentracion de casos   : {med_carga:.1f} % del total nacional")
say("\n--- RANKING DEPARTAMENTAL ---")
say(f"{'#':>2} {'Departamento':16s} {'Prev':>6} {'Pred':>6} {'% casos':>8}  Cuadrante")
for _, r in dep.iterrows():
    say(f"{r['ranking']:>2} {r['departamento']:16s} {r['prevalencia_obs']:5.1f}% "
        f"{r['riesgo_predicho']:5.1f}% {r['pct_casos_pais']:7.1f}%  {r['cuadrante']}")

say("\n--- CUADRANTES DE PRIORIZACION ---")
for c, g in dep.groupby("cuadrante"):
    say(f"\n   {c}  ({len(g)} departamentos, {g.pct_casos_pais.sum():.1f} % de los casos del pais)")
    say("      " + ", ".join(g.sort_values('prevalencia_obs', ascending=False).departamento))

# --------------------------------------------------------------------------
# 2. Dominio: departamento x area
# --------------------------------------------------------------------------
dom = agregar(p, ["departamento", "area"]).sort_values("prevalencia_obs", ascending=False)
dom = dom[dom.n_muestra >= 50]
dom.to_csv(os.path.join(OUT, "riesgo_dominio.csv"), index=False, encoding="utf-8-sig")
say("\n--- TOP 10 DOMINIOS (departamento x area, n>=50) ---")
for _, r in dom.head(10).iterrows():
    say(f"   {r['departamento']:16s} {r['area']:8s} {r['prevalencia_obs']:5.1f} %   "
        f"(n={int(r['n_muestra'])})")

# --------------------------------------------------------------------------
# 3. Region natural y altitud
# --------------------------------------------------------------------------
agregar(p, "region_natural").sort_values("prevalencia_obs", ascending=False) \
    .to_csv(os.path.join(OUT, "riesgo_region_natural.csv"), index=False, encoding="utf-8-sig")

p["altitud_tramo"] = pd.cut(p.altitud_m, [-100, 499, 999, 1999, 2999, 3499, 3999, 6000],
                            labels=["0-499", "500-999", "1000-1999", "2000-2999",
                                    "3000-3499", "3500-3999", "4000+"])
alt = agregar(p, "altitud_tramo")
alt.to_csv(os.path.join(OUT, "riesgo_altitud.csv"), index=False, encoding="utf-8-sig")
say("\n--- GRADIENTE DE ALTITUD (tramos de 500 a 1000 m) ---")
for _, r in alt.iterrows():
    say(f"   {str(r['altitud_tramo']):12s} {r['prevalencia_obs']:5.1f} %   (n={int(r['n_muestra']):,})")

# --------------------------------------------------------------------------
# 4. Conglomerados: el mapa de puntos para Power BI
# --------------------------------------------------------------------------
cong = p.groupby("conglomerado").apply(lambda x: pd.Series({
    "departamento": x.departamento.iloc[0],
    "area": x.area.iloc[0],
    "latitud": x.latitud.mean(),
    "longitud": x.longitud.mean(),
    "altitud_m": x.altitud_m.mean(),
    "n_ninos": len(x),
    "riesgo_predicho": x.riesgo_predicho.mean() * 100,
    "prevalencia_obs": x.anemia.mean() * 100,
}), include_groups=False).reset_index()
cong = cong[cong.n_ninos >= 3].round(3)
cong.to_csv(os.path.join(OUT, "riesgo_conglomerado.csv"), index=False, encoding="utf-8-sig")
say(f"\nConglomerados georreferenciados exportados: {len(cong):,}")

# --------------------------------------------------------------------------
# 5. Perfil del nino de maximo riesgo
# --------------------------------------------------------------------------
say("\n--- PERFIL DEL DECIL DE MAYOR RIESGO vs EL DE MENOR RIESGO ---")
d10 = p[p.decil_riesgo == 10]
d1 = p[p.decil_riesgo == 1]
say(f"   Prevalencia real en el decil 10: {np.average(d10.anemia, weights=d10.factor)*100:.1f} %")
say(f"   Prevalencia real en el decil  1: {np.average(d1.anemia, weights=d1.factor)*100:.1f} %")
bb = b.set_index("id_nino")
for col, etq, fmt in [("edad_meses", "Edad (meses)", "{:.1f}"),
                      ("altitud_m", "Altitud (m)", "{:.0f}"),
                      ("quintil_riqueza", "Quintil de riqueza", "{:.1f}"),
                      ("hb_madre", "Hemoglobina de la madre", "{:.1f}"),
                      ("talla_cm", "Talla (cm)", "{:.1f}")]:
    v10 = bb.loc[d10.id_nino, col].mean()
    v1 = bb.loc[d1.id_nino, col].mean()
    say(f"   {etq:26s} decil 10: {fmt.format(v10):>7}   decil 1: {fmt.format(v1):>7}")

perfil = pd.DataFrame([{
    "variable": etq,
    "decil_10_mayor_riesgo": round(float(bb.loc[d10.id_nino, col].mean()), 2),
    "decil_1_menor_riesgo": round(float(bb.loc[d1.id_nino, col].mean()), 2)}
    for col, etq in [("edad_meses", "Edad en meses"), ("altitud_m", "Altitud en metros"),
                     ("quintil_riqueza", "Quintil de riqueza"),
                     ("hb_madre", "Hemoglobina de la madre"), ("talla_cm", "Talla en cm"),
                     ("tam_hogar", "Tamano del hogar")]])
perfil.to_csv(os.path.join(OUT, "perfil_deciles.csv"), index=False, encoding="utf-8-sig")

resumen = {
    "mediana_prevalencia_dep": round(float(med_prev), 1),
    "departamentos_criticos": dep[dep.cuadrante.str.startswith("1")].departamento.tolist(),
    "pct_casos_criticos": round(float(dep[dep.cuadrante.str.startswith("1")].pct_casos_pais.sum()), 1),
    "top5": dep.head(5)[["departamento", "prevalencia_obs", "pct_casos_pais"]].to_dict("records"),
    "concentracion_top5_casos": round(float(dep.nlargest(5, "pct_casos_pais").pct_casos_pais.sum()), 1),
    "gradiente_altitud": alt[["altitud_tramo", "prevalencia_obs", "n_muestra"]]
                          .astype({"altitud_tramo": str}).to_dict("records"),
}
with open(os.path.join(BASE, "Resultados", "resumen_mapa.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2, default=float)

with open(os.path.join(OUT, "reporte_mapa.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))

say(f"\nSalidas en {OUT}")
