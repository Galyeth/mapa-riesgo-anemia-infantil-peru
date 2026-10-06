"""
05 - Figuras del estudio (PNG listos para el informe).

Entrada : Resultados/tablas/*, Resultados/modelo/*, Resultados/mapa/*
Salidas : Resultados/figuras/*.png

Criterios de diseño: una sola serie por grafico (el color no codifica identidad,
solo resalta), marcas delgadas, sin rejilla pesada, etiquetas directas en vez de
leyendas, y acento rojo reservado para señalar lo critico.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("PROY_DIR", ".")
R = os.path.join(BASE, "Resultados")
F = os.path.join(R, "figuras")
os.makedirs(F, exist_ok=True)

AZUL = "#2a78d6"
AZUL_SUAVE = "#cde2fb"
ROJO = "#e34948"
TINTA = "#1a1a19"
TINTA2 = "#52514e"
SUP = "#fcfcfb"
GRIS = "#d9d8d4"

plt.rcParams.update({
    "figure.facecolor": SUP, "axes.facecolor": SUP,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.titlecolor": TINTA, "axes.titlesize": 12.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14,
})


def limpiar(ax, ejes=("top", "right", "left")):
    for e in ejes:
        ax.spines[e].set_visible(False)
    ax.tick_params(length=0)


def guardar(fig, nombre):
    fig.savefig(os.path.join(F, nombre), dpi=170, bbox_inches="tight",
                facecolor=SUP)
    plt.close(fig)
    print("  ->", nombre)


# --------------------------------------------------------------- F1 edad
t = pd.read_csv(os.path.join(R, "tablas", "prevalencia_edad.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.4, 4.1))
cols = [ROJO if v == t.prevalencia_anemia.max() else AZUL for v in t.prevalencia_anemia]
b = ax.bar(t.grupo_edad, t.prevalencia_anemia, color=cols, width=0.62)
for x, v in zip(range(len(t)), t.prevalencia_anemia):
    ax.text(x, v + 1.6, f"{v:.1f}%", ha="center", color=TINTA, fontsize=10.5, fontweight="bold")
ax.set_ylim(0, 75)
ax.set_ylabel("Prevalencia de anemia")
ax.set_title("La anemia se concentra en el primer año de vida")
ax.set_yticks([0, 20, 40, 60])
ax.set_yticklabels(["0", "20%", "40%", "60%"])
ax.yaxis.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax)
guardar(fig, "f1_prevalencia_por_edad.png")

# ----------------------------------------------------------- F2 altitud
t = pd.read_csv(os.path.join(R, "mapa", "riesgo_altitud.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.4, 4.1))
cols = [ROJO if v >= 55 else AZUL for v in t.prevalencia_obs]
ax.bar(t.altitud_tramo.astype(str), t.prevalencia_obs, color=cols, width=0.62)
for x, v in zip(range(len(t)), t.prevalencia_obs):
    ax.text(x, v + 1.3, f"{v:.1f}%", ha="center", color=TINTA, fontsize=10, fontweight="bold")
ax.set_ylim(0, 70)
ax.set_yticks([0, 20, 40, 60])
ax.set_yticklabels(["0", "20%", "40%", "60%"])
ax.set_xlabel("Altitud de residencia (metros sobre el nivel del mar)")
ax.set_ylabel("Prevalencia de anemia")
ax.set_title("Por encima de los 3 000 metros la prevalencia se duplica")
ax.yaxis.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax)
guardar(fig, "f2_gradiente_altitud.png")

# ------------------------------------------------------- F3 departamentos
d = pd.read_csv(os.path.join(R, "mapa", "riesgo_departamento.csv"), encoding="utf-8-sig")
d = d.sort_values("prevalencia_obs")
fig, ax = plt.subplots(figsize=(7.4, 8.2))
cols = [ROJO if c.startswith("1.") else AZUL for c in d.cuadrante]
ax.barh(d.departamento, d.prevalencia_obs, color=cols, height=0.66)
for y, v in enumerate(d.prevalencia_obs):
    ax.text(v + 0.8, y, f"{v:.1f}%", va="center", color=TINTA, fontsize=9.5)
ax.axvline(33.6, color=TINTA2, lw=1.2, ls=(0, (4, 3)))
ax.text(34.4, -1.3, "Promedio nacional 33.6%", color=TINTA2, fontsize=9)
ax.set_xlim(0, 72)
ax.set_xticks([0, 20, 40, 60])
ax.set_xticklabels(["0", "20%", "40%", "60%"])
ax.set_title("Puno triplica la prevalencia de Tacna\nEn rojo, los departamentos del cuadrante crítico")
ax.xaxis.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax, ("top", "right", "left", "bottom"))
guardar(fig, "f3_ranking_departamentos.png")

# -------------------------------------------------------- F4 cuadrantes
fig, ax = plt.subplots(figsize=(8.2, 5.8))
mx = d.prevalencia_obs.median()
my = d.pct_casos_pais.median()
ax.set_xlim(18, 70)
ax.set_ylim(-1.2, 21.5)
ax.axvline(mx, color=GRIS, lw=1.4)
ax.axhline(my, color=GRIS, lw=1.4)
ax.scatter(d.prevalencia_obs, d.pct_casos_pais, s=86, color=AZUL,
           edgecolor=SUP, linewidth=2, zorder=3)

OFF = {"Puno": (-12, 12), "Loreto": (10, 2), "Lima": (10, -2), "Cusco": (9, -4),
       "Tacna": (9, 2), "La Libertad": (9, 3), "Junin": (9, 2),
       "Madre de Dios": (-18, -16), "Huancavelica": (10, -3),
       "Ucayali": (9, 3), "Apurimac": (-30, 8)}
for _, r in d.iterrows():
    if r.departamento in OFF:
        ax.annotate(r.departamento, (r.prevalencia_obs, r.pct_casos_pais),
                    textcoords="offset points", xytext=OFF[r.departamento],
                    fontsize=9.5, color=TINTA, fontweight="bold")

ax.text(69, 20.8, "CRÍTICO", fontsize=9.5, color=ROJO, fontweight="bold",
        ha="right", va="top")
ax.text(69, 19.4, "alta prevalencia + alto volumen", fontsize=8.5,
        color=TINTA2, ha="right", va="top")
ax.text(19, 20.8, "VOLUMEN", fontsize=9.5, color=TINTA2, fontweight="bold",
        ha="left", va="top")
ax.text(19, 19.4, "prevalencia media, muchos niños", fontsize=8.5,
        color=TINTA2, ha="left", va="top")
ax.text(69, -0.3, "FOCALIZADO", fontsize=9.5, color=TINTA2, fontweight="bold",
        ha="right", va="bottom")
ax.text(69, -1.1, "alta prevalencia, pocos niños", fontsize=8.5,
        color=TINTA2, ha="right", va="bottom")
ax.text(19, -0.3, "VIGILANCIA", fontsize=9.5, color=TINTA2, fontweight="bold",
        ha="left", va="bottom")

ax.set_xlabel("Prevalencia de anemia en el departamento")
ax.set_ylabel("Porcentaje de todos los casos del país")
ax.set_title("Dónde es más grave no es dónde hay más niños")
ax.set_xticks([20, 30, 40, 50, 60, 70])
ax.set_xticklabels(["20%", "30%", "40%", "50%", "60%", "70%"])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_yticklabels(["0", "5%", "10%", "15%", "20%"])
limpiar(ax, ("top", "right"))
guardar(fig, "f4_cuadrantes_priorizacion.png")

# ------------------------------------------------------- F5 calibracion
c = pd.read_csv(os.path.join(R, "modelo", "calibracion.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, .85], [0, .85], color=GRIS, lw=1.6, ls=(0, (4, 3)), zorder=1)
ax.plot(c.riesgo_predicho_promedio, c.prevalencia_observada, color=AZUL, lw=2,
        marker="o", ms=8, mfc=AZUL, mec=SUP, mew=2, zorder=3)
ax.text(.60, .50, "calibración perfecta", color=TINTA2, fontsize=9, rotation=38)
ax.set_xlabel("Riesgo predicho por el modelo")
ax.set_ylabel("Prevalencia realmente observada")
ax.set_title("El modelo dice la verdad sobre su\npropia incertidumbre")
ax.set_xlim(0, .85)
ax.set_ylim(0, .85)
ax.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax, ("top", "right"))
guardar(fig, "f5_calibracion.png")

# -------------------------------------------------------- F6 importancia
i = pd.read_csv(os.path.join(R, "modelo", "importancia_variables.csv"), encoding="utf-8-sig")
ETQ = {"talla_cm": "Talla del niño", "hb_madre": "Hemoglobina de la madre",
       "edad_meses": "Edad en meses", "altitud_m": "Altitud de residencia",
       "indice_riqueza": "Índice de riqueza", "longitud": "Longitud geográfica",
       "sexo": "Sexo del niño", "peso_kg": "Peso del niño",
       "intervalo_nacimiento_meses": "Intervalo entre nacimientos",
       "latitud": "Latitud geográfica", "edad_madre": "Edad de la madre",
       "z_peso_talla": "Z peso para la talla", "anemia_madre": "Anemia de la madre",
       "z_talla_edad": "Z talla para la edad", "imc_madre": "IMC de la madre"}
top = i.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.4, 5.4))
ax.barh([ETQ.get(v, v) for v in top.variable], top.importancia_relativa,
        color=AZUL, height=0.64)
for y, v in enumerate(top.importancia_relativa):
    ax.text(v + 0.25, y, f"{v:.1f}%", va="center", color=TINTA, fontsize=9.5)
ax.set_xlabel("Contribución a la predicción (SHAP, % del total)")
ax.set_title("Qué mira el modelo para decidir")
ax.xaxis.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax, ("top", "right", "left", "bottom"))
guardar(fig, "f6_importancia_variables.png")

# ---------------------------------------------------------- F7 utilidad
u = pd.read_csv(os.path.join(R, "modelo", "utilidad_operativa.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = np.concatenate([[0], u.fraccion_visitada * 100])
y = np.concatenate([[0], u.sensibilidad * 100])
ax.plot([0, 100], [0, 100], color=GRIS, lw=1.6, ls=(0, (4, 3)))
ax.plot(x, y, color=AZUL, lw=2.4, marker="o", ms=7, mfc=AZUL, mec=SUP, mew=2)
ax.fill_between(x, np.linspace(0, 50, len(x)) * 0 + x, y, color=AZUL, alpha=.07)
ax.annotate("Visitando al 20% más riesgoso\nse alcanza al 39% de los casos",
            (20, 38.7), textcoords="offset points", xytext=(14, -34),
            fontsize=9.5, color=TINTA,
            arrowprops=dict(arrowstyle="-", color=TINTA2, lw=1))
ax.text(62, 56, "visita al azar", color=TINTA2, fontsize=9, rotation=30)
ax.set_xlabel("Porcentaje de niños visitados, priorizando por riesgo")
ax.set_ylabel("Porcentaje de casos de anemia detectados")
ax.set_title("Para qué sirve el modelo en campo")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax, ("top", "right"))
guardar(fig, "f7_utilidad_operativa.png")

# ----------------------------------------------------------- F8 riqueza
t = pd.read_csv(os.path.join(R, "tablas", "prevalencia_quintil.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.bar(t.quintil_riqueza_lbl, t.prevalencia_anemia, color=AZUL, width=0.6)
for x_, v in zip(range(len(t)), t.prevalencia_anemia):
    ax.text(x_, v + 1.1, f"{v:.1f}%", ha="center", color=TINTA, fontsize=10, fontweight="bold")
ax.set_ylim(0, 52)
ax.set_yticks([0, 20, 40])
ax.set_yticklabels(["0", "20%", "40%"])
ax.set_xlabel("Quintil de riqueza del hogar")
ax.set_title("El gradiente socioeconómico es continuo, sin saltos")
ax.yaxis.grid(True, color=GRIS, lw=0.7)
ax.set_axisbelow(True)
limpiar(ax)
guardar(fig, "f8_quintil_riqueza.png")

print(f"\nFiguras en {F}")
