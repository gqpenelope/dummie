import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sco
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolios", layout="wide")
st.title("Análisis y Optimización de Portafolios")

# Definición de ETFs y ventanas de tiempo
etfs = ['LQD', 'EMB', 'ACWI', 'SPY', 'WMT']
ventanas = {
    "2010-2023": ("2010-01-01", "2023-12-31"),
    "2010-2020": ("2010-01-01", "2020-12-31"),
    "2021-2023": ("2021-01-01", "2023-12-31")
}

# Función para descargar datos de Yahoo Finance
@st.cache
def obtener_datos(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Close']
    return data

# Selección de ventana de tiempo
st.sidebar.header("Configuración de Ventana")
ventana = st.sidebar.selectbox("Selecciona una ventana de tiempo:", list(ventanas.keys()))
start_date, end_date = ventanas[ventana]

# Descarga de datos
datos = obtener_datos(etfs, start_date, end_date)
rendimientos = datos.pct_change().dropna()

# Cálculo de métricas
def calcular_metricas(rendimientos):
    media = rendimientos.mean() * 252
    volatilidad = rendimientos.std() * np.sqrt(252)
    sharpe = media / volatilidad
    sesgo = rendimientos.skew()
    curtosis = rendimientos.kurt()
    drawdown = (rendimientos.cumsum() - rendimientos.cumsum().cummax()).min()
    var = rendimientos.quantile(0.05)
    cvar = rendimientos[rendimientos <= var].mean()
    return {
        "Media": media,
        "Volatilidad": volatilidad,
        "Sharpe": sharpe,
        "Sesgo": sesgo,
        "Curtosis": curtosis,
        "Drawdown": drawdown,
        "VaR": var,
        "CVaR": cvar
    }

metricas = {etf: calcular_metricas(rendimientos[etf]) for etf in etfs}
metricas_df = pd.DataFrame(metricas).T

# Visualización de métricas
st.header(f"Estadísticas para la ventana {ventana}")
st.dataframe(metricas_df)

# Optimización de portafolios
def optimizar_portafolio(rendimientos, objetivo="sharpe", rendimiento_objetivo=None):
    media = rendimientos.mean() * 252
    covarianza = rendimientos.cov() * 252
    num_activos = len(media)
    pesos_iniciales = np.ones(num_activos) / num_activos
    limites = [(0, 1) for _ in range(num_activos)]
    restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if objetivo == "sharpe":
        def objetivo_func(pesos):
            rendimiento = np.dot(pesos, media)
            riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
            return -rendimiento / riesgo
    elif objetivo == "volatilidad":
        def objetivo_func(pesos):
            return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
    elif objetivo == "rendimiento":
        restricciones.append({'type': 'eq', 'fun': lambda x: np.dot(x, media) - rendimiento_objetivo})
        def objetivo_func(pesos):
            return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

    resultado = sco.minimize(objetivo_func, pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado.x

# Pesos óptimos para portafolios
pesos_sharpe = optimizar_portafolio(rendimientos, objetivo="sharpe")
pesos_volatilidad = optimizar_portafolio(rendimientos, objetivo="volatilidad")
pesos_rendimiento = optimizar_portafolio(rendimientos, objetivo="rendimiento", rendimiento_objetivo=0.10)

pesos_df = pd.DataFrame({
    "Máximo Sharpe": pesos_sharpe,
    "Mínima Volatilidad": pesos_volatilidad,
    "Rendimiento Objetivo 10%": pesos_rendimiento
}, index=etfs)

st.header("Pesos de Portafolios Óptimos")
st.bar_chart(pesos_df)

# Gráficos de precios normalizados
precios_normalizados = datos / datos.iloc[0] * 100
fig = go.Figure()
for etf in etfs:
    fig.add_trace(go.Scatter(x=precios_normalizados.index, y=precios_normalizados[etf], mode='lines', name=etf))
fig.update_layout(title="Precios Normalizados", xaxis_title="Fecha", yaxis_title="Precio Normalizado")
st.plotly_chart(fig)

# Información descriptiva de activos
st.sidebar.header("Descripción de Activos")
descripciones = {
    "LQD": "Bonos corporativos denominados en USD con grado de inversión.",
    "EMB": "Bonos de mercados emergentes denominados en USD.",
    "ACWI": "Empresas internacionales de mercados desarrollados y emergentes.",
    "SPY": "Empresas de alta capitalización de Estados Unidos.",
    "WMT": "Retailer global con enfoque en mercados de Estados Unidos."
}
activo_seleccionado = st.sidebar.selectbox("Selecciona un activo:", etfs)
st.sidebar.write(descripciones[activo_seleccionado])
