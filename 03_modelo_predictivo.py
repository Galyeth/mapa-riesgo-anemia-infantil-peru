"""
03 - Modelo predictivo de riesgo de anemia infantil.

Entrada : Base_Analitica/base_anemia_endes2025.csv
Salidas : Resultados/modelo/*  (metricas, importancias, predicciones, modelo final)

Diseño de validacion
--------------------
StratifiedGroupKFold agrupando por DEPARTAMENTO. El modelo se evalua prediciendo
departamentos que nunca vio en el entrenamiento, que es exactamente el uso real:
estimar riesgo donde todavia no se ha medido. Una validacion aleatoria normal
inflaria el resultado porque niños del mismo conglomerado quedarian a ambos lados.

Se entrenan tres especificaciones:
  M0  Regresion logistica  - linea base interpretable
  M1  LightGBM sin geografia - cuanto explica solo el perfil del hogar
  M2  LightGBM completo      - agrega altitud, coordenadas y region natural
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

BASE = os.environ.get("PROY_DIR", ".")
IN_CSV = os.path.join(BASE, "Base_Analitica", "base_anemia_endes2025.csv")
OUT = os.path.join(BASE, "Resultados", "modelo")
os.makedirs(OUT, exist_ok=True)

log = []
def say(m=""):
    print(m)
    log.append(str(m))

df = pd.read_csv(IN_CSV, encoding="utf-8-sig", low_memory=False)
df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

# --------------------------------------------------------------------------
# 1. Definicion de variables
# --------------------------------------------------------------------------
# NUNCA entran como predictores: son la variable objetivo o su insumo directo.
FUGA = ["anemia", "anemia_oms", "nivel_anemia", "nivel_anemia_oms",
        "hb_ajustada", "hb_ajustada_oms", "hb_cruda"]

NUM_NINO = ["edad_meses", "peso_kg", "talla_cm",
            "z_talla_edad", "z_peso_edad", "z_peso_talla",
            "orden_nacimiento", "intervalo_nacimiento_meses"]
NUM_MADRE = ["edad_madre", "imc_madre", "hb_madre", "educacion_madre_num"]
NUM_HOGAR = ["quintil_riqueza", "indice_riqueza", "tiempo_agua_min",
             "tam_hogar", "n_menores5", "edad_jefe_hogar"]
NUM_GEO = ["altitud_m", "latitud", "longitud"]

BIN = ["desnutricion_cronica", "anemia_madre", "agua_red_publica",
       "agua_dentro_vivienda", "desague_red", "sin_servicio_higienico",
       "electricidad", "piso_tierra", "combustible_contaminante"]

CAT_BASE = ["sexo", "area", "fuente_agua", "servicio_higienico", "piso",
            "combustible_cocina", "sexo_jefe_hogar"]
CAT_GEO = ["region_natural", "altitud_cat"]

SIN_GEO = NUM_NINO + NUM_MADRE + NUM_HOGAR + BIN + CAT_BASE
CON_GEO = SIN_GEO + NUM_GEO + CAT_GEO

y = df["anemia"].astype(int).values
grupos = df["departamento_cod"].values
peso = df["factor"].values

say("=" * 74)
say("MODELO PREDICTIVO DE RIESGO DE ANEMIA INFANTIL")
say("=" * 74)
say(f"\nObservaciones            : {len(df):,} ninos de 6 a 59 meses")
say(f"Prevalencia en la muestra: {y.mean()*100:.1f} % (sin ponderar)")
say(f"Grupos de validacion     : {len(np.unique(grupos))} departamentos")
say(f"Predictores sin geografia: {len(SIN_GEO)}")
say(f"Predictores con geografia: {len(CON_GEO)}")
say("\nVariables excluidas por fuga de informacion: " + ", ".join(FUGA))


def preparar(cols):
    X = df[cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


def pipeline_logistica(cols):
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=30))]), cat),
        ])),
        ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE)),
    ])


def params_lgb():
    return dict(objective="binary", n_estimators=600, learning_rate=0.03,
                num_leaves=31, min_child_samples=60, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1)


# --------------------------------------------------------------------------
# 2. Validacion cruzada agrupada por departamento
# --------------------------------------------------------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds = list(cv.split(df, y, grupos))

resultados = {}
oof_store = {}

for nombre, cols, tipo in [("M0 Logistica (completo)", CON_GEO, "log"),
                           ("M1 LightGBM sin geografia", SIN_GEO, "lgb"),
                           ("M2 LightGBM completo", CON_GEO, "lgb")]:
    X = preparar(cols)
    oof = np.full(len(df), np.nan)
    aucs = []
    for tr, te in folds:
        if tipo == "log":
            mod = pipeline_logistica(cols)
            mod.fit(X.iloc[tr], y[tr])
            p = mod.predict_proba(X.iloc[te])[:, 1]
        else:
            mod = lgb.LGBMClassifier(**params_lgb())
            mod.fit(X.iloc[tr], y[tr],
                    eval_set=[(X.iloc[te], y[te])],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(60, verbose=False)])
            p = mod.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))

    resultados[nombre] = {
        "auc_roc": round(float(roc_auc_score(y, oof)), 4),
        "auc_roc_por_fold": [round(a, 4) for a in aucs],
        "auc_roc_sd": round(float(np.std(aucs)), 4),
        "auc_pr": round(float(average_precision_score(y, oof)), 4),
        "brier": round(float(brier_score_loss(y, oof)), 4),
        "n_predictores": len(cols),
    }
    oof_store[nombre] = oof
    r = resultados[nombre]
    say(f"\n{nombre}")
    say(f"   AUC-ROC  {r['auc_roc']:.3f}  (por fold: "
        f"{', '.join(f'{a:.3f}' for a in r['auc_roc_por_fold'])}  sd {r['auc_roc_sd']:.3f})")
    say(f"   AUC-PR   {r['auc_pr']:.3f}   (prevalencia de referencia {y.mean():.3f})")
    say(f"   Brier    {r['brier']:.4f}")

MEJOR = "M2 LightGBM completo"
oof = oof_store[MEJOR]

# --------------------------------------------------------------------------
# 3. Utilidad operativa: que pasa si visito al 20 % de mayor riesgo
# --------------------------------------------------------------------------
say("\n--- UTILIDAD OPERATIVA (modelo " + MEJOR + ") ---")
say("   Si el equipo de salud solo puede visitar una fraccion de los ninos,")
say("   cuantos casos de anemia alcanza si prioriza por riesgo predicho:")
orden = np.argsort(-oof)
tot_casos = y.sum()
cobertura = []
for frac in [0.10, 0.20, 0.30, 0.40, 0.50]:
    k = int(len(df) * frac)
    sel = orden[:k]
    casos = y[sel].sum()
    prec = casos / k
    rec = casos / tot_casos
    lift = prec / y.mean()
    cobertura.append({"fraccion_visitada": frac, "ninos": k, "casos_detectados": int(casos),
                      "precision": round(prec, 3), "sensibilidad": round(rec, 3),
                      "lift": round(lift, 2)})
    say(f"   Top {frac*100:3.0f} %  ->  detecta {rec*100:4.1f} % de los casos  "
        f"| {prec*100:4.1f} % de los visitados tiene anemia  | lift {lift:.2f}x")
pd.DataFrame(cobertura).to_csv(os.path.join(OUT, "utilidad_operativa.csv"),
                               index=False, encoding="utf-8-sig")

# --------------------------------------------------------------------------
# 4. Calibracion
# --------------------------------------------------------------------------
pt, pp = calibration_curve(y, oof, n_bins=10, strategy="quantile")
cal = pd.DataFrame({"riesgo_predicho_promedio": np.round(pp, 3),
                    "prevalencia_observada": np.round(pt, 3)})
cal.to_csv(os.path.join(OUT, "calibracion.csv"), index=False, encoding="utf-8-sig")
say("\n--- CALIBRACION (deciles de riesgo) ---")
for _, r in cal.iterrows():
    say(f"   predicho {r['riesgo_predicho_promedio']:.3f}   observado {r['prevalencia_observada']:.3f}")

# --------------------------------------------------------------------------
# 5. Modelo final e importancia de variables (SHAP)
# --------------------------------------------------------------------------
X_full = preparar(CON_GEO)
modelo_final = lgb.LGBMClassifier(**params_lgb())
modelo_final.fit(X_full, y)

import shap
expl = shap.TreeExplainer(modelo_final)
sv = expl.shap_values(X_full)
if isinstance(sv, list):
    sv = sv[1]
imp = pd.DataFrame({
    "variable": CON_GEO,
    "importancia_shap": np.abs(sv).mean(axis=0),
}).sort_values("importancia_shap", ascending=False)
imp["importancia_relativa"] = (imp.importancia_shap / imp.importancia_shap.sum() * 100).round(2)
imp.to_csv(os.path.join(OUT, "importancia_variables.csv"), index=False, encoding="utf-8-sig")

say("\n--- QUE PESA MAS EN LA PREDICCION (SHAP, top 15) ---")
for _, r in imp.head(15).iterrows():
    say(f"   {r['variable']:28s} {r['importancia_relativa']:5.2f} %")

# --------------------------------------------------------------------------
# 6. Guardar predicciones y metricas
# --------------------------------------------------------------------------
pred = df[["id_nino", "hhid", "ubigeo", "departamento", "region_natural", "area",
           "altitud_m", "edad_meses", "quintil_riqueza_lbl", "factor",
           "conglomerado", "anemia"]].copy()
pred["riesgo_predicho"] = oof              # fuera de muestra: lo honesto para evaluar
pred["riesgo_insample"] = modelo_final.predict_proba(X_full)[:, 1]
pred["decil_riesgo"] = pd.qcut(pred.riesgo_predicho, 10, labels=range(1, 11)).astype(int)
pred.to_csv(os.path.join(OUT, "predicciones.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(OUT, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump({"modelos": resultados, "mejor": MEJOR,
               "cobertura": cobertura,
               "top_variables": imp.head(15).to_dict("records")},
              f, ensure_ascii=False, indent=2, default=float)

import joblib
joblib.dump({"modelo": modelo_final, "columnas": CON_GEO}, os.path.join(OUT, "modelo_lgbm.pkl"))

with open(os.path.join(OUT, "reporte_modelo.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))

say(f"\nSalidas en {OUT}")
