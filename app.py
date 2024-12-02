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
