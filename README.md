# Mapa de riesgo de anemia infantil en el Perú
### Proyecto de Data Science - ENDES 2025 (INEI)

## Cómo correr el proyecto

Desde la carpeta `Proyecto`, en orden:

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib joblib

python 01_construir_base.py        # arma la base analítica desde los CSV del INEI
python 02_analisis_exploratorio.py # prevalencias ponderadas y tablas
python 03_modelo_predictivo.py     # entrena y valida los tres modelos
python 04_mapa_riesgo.py           # agrega el riesgo por territorio
python 05_figuras.py               # genera las figuras del informe
```

Los scripts leen y escriben con rutas relativas. No modifican ningún archivo
original del INEI: todo lo generado vive en `Base_Analitica/` y `Resultados/`.

## Estructura

```
Proyecto/
├── 1038-Modulo1629/          datos originales del INEI (no tocar)
├── 1038-Modulo1630/
├── 1038-Modulo1638/
├── Base_Analitica/
│   ├── base_anemia_endes2025.csv     17 616 niños × 60 variables
│   ├── base_anemia_endes2025.db      la misma tabla en SQLite
│   └── diccionario_base_anemia.csv   qué es cada variable
└── Resultados/
    ├── tablas/      prevalencias por cada corte
    ├── modelo/      métricas, importancias, predicciones, modelo entrenado
    ├── mapa/        riesgo por departamento, dominio, altitud y conglomerado
    └── figuras/     las 8 figuras del informe en PNG
```

## Fuente

Encuesta Demográfica y de Salud Familiar (ENDES) 2025, Instituto Nacional de
Estadística e Informática del Perú. Módulos 1629, 1630 y 1638.
Microdatos de acceso público: https://proyectos.inei.gob.pe/microdatos/

## Para Power BI

Conectar a `Resultados/mapa/riesgo_departamento.csv` para el mapa coroplético
y a `riesgo_conglomerado.csv` para el mapa de puntos georreferenciados.
