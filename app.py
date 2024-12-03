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

# Selección de ventana de tiempo en el sidebar (solo una vez)
st.sidebar.header("Configuración de Ventana")
ventana = st.sidebar.selectbox(
    "Selecciona una ventana de tiempo para análisis:",
    options=list(ventanas.keys())
)
start_date, end_date = ventanas[ventana]

# Función para descargar datos de Yahoo Finance
@st.cache_data
def obtener_datos(etfs, start_date, end_date):
    try:
        data = yf.download(etfs, start=start_date, end=end_date)['Close']
        if data.empty:
            st.error(f"No se encontraron datos para los ETFs: {etfs} entre {start_date} y {end_date}.")
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()

# Descarga de datos
datos = obtener_datos(etfs, start_date, end_date)

# Validar datos
if datos.empty:
    st.error("No se encontraron datos para los parámetros seleccionados.")
else:
    # Mostrar datos descargados
    st.subheader(f"Datos descargados para la ventana {ventana}")
    st.dataframe(datos.head())

    # Calcular rendimientos
    rendimientos = datos.pct_change().dropna()

    if rendimientos.empty:
        st.error("No se pudieron calcular los rendimientos.")
    else:
        st.write("Rendimientos calculados correctamente.")

        # Continuar con análisis y optimización
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

# Descripción detallada de cada ETF
descripciones_etfs = {
    "LQD": {
        "nombre": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "exposicion": "Bonos corporativos denominados en dólares de EE.UU. y con grado de inversión.",
        "indice": "iBoxx $ Liquid Investment Grade Index",
        "moneda": "USD",
        "principales": ["JPMorgan Chase & Co", "Bank of America Corp", "Morgan Stanley"],
        "paises": "Estados Unidos y empresas multinacionales",
        "metrica_riesgo": {
            "Rendimiento 12 meses": "4.36%",
            "Vencimiento promedio": "13.21 años",
            "Desviación estándar (3 años)": "11.49",
            "Convexidad": "1.22",
            "Duración": "8.41 años",
            "Beta a 3 años": "0.53",
            "Cupón promedio": "4.32"
        },
        "estilo": "Value",
        "costos": "Comisión de administración: 0.14%"
    },
    "EMB": {
        "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "exposicion": "Bonos de gobierno denominados en USD emitidos por países de mercados emergentes.",
        "indice": "JPMorgan EMBI Global Core Index",
        "moneda": "USD",
        "principales": ["Turkey (Republic of)", "Saudi Arabia (Kingdom of)", "Brazil Federative Republic of"],
        "paises": "América Latina, Medio Oriente, África y Asia",
        "metrica_riesgo": {
            "Rendimiento 12 meses": "4.91%",
            "Vencimiento promedio": "11.72 años",
            "Desviación estándar (3 años)": "11.62%",
            "Convexidad": "0.86",
            "Duración": "7.01 años",
            "Beta a 3 años": "0.57",
            "Cupón promedio": "5.21"
        },
        "estilo": "Value",
        "costos": "Comisión de administración: 0.39%"
    },
    "ACWI": {
        "nombre": "iShares MSCI ACWI ETF",
        "exposicion": "Empresas internacionales de mercados desarrollados y emergentes de alta y mediana capitalización.",
        "indice": "MSCI ACWI Index",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos y mercados desarrollados/emergentes",
        "metrica_riesgo": {
            "Rendimiento 12 meses": "1.62%",
            "Desviación estándar (3 años)": "16.67%",
            "Beta a 3 años": "0.95"
        },
        "estilo": "Growth",
        "costos": "Comisión de administración: 0.32%"
    },
    "SPY": {
        "nombre": "iShares Core S&P 500 ETF",
        "exposicion": "Empresas de alta capitalización en Estados Unidos.",
        "indice": "S&P 500 Index (USD)",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos",
        "metrica_riesgo": {
            "Rendimiento 12 meses": "1.29%",
            "Desviación estándar (3 años)": "17.20%",
            "Beta a 3 años": "1.00"
        },
        "estilo": "Mix (Growth/Value)",
        "costos": "Comisión de administración: 0.03%"
    },
    "WMT": {
        "nombre": "Walmart Inc",
        "exposicion": "Retailer global enfocado en mercados de Estados Unidos.",
        "indice": "N/A",
        "moneda": "USD",
        "principales": ["N/A"],
        "paises": "Estados Unidos y mercados internacionales",
        "metrica_riesgo": {
            "Rendimiento 12 meses": "N/A",
            "Desviación estándar (3 años)": "N/A",
            "Beta a 3 años": "N/A"
        },
        "estilo": "Mix (Growth/Value)",
        "costos": "N/A"
    }
}

# Pestañas interactivas para descripción de ETFs
tabs = st.tabs(list(descripciones_etfs.keys()))
for i, etf in enumerate(descripciones_etfs.keys()):
    with tabs[i]:
        data = descripciones_etfs[etf]
        st.header(data["nombre"])
        st.subheader("Exposición")
        st.write(data["exposicion"])
        st.subheader("Índice")
        st.write(data["indice"])
        st.subheader("Moneda")
        st.write(data["moneda"])
        st.subheader("Principales contribuyentes")
        st.write(", ".join(data["principales"]))
        st.subheader("Países")
        st.write(data["paises"])
        st.subheader("Métricas de Riesgo")
        st.table(pd.DataFrame(data["metrica_riesgo"], index=["Valores"]).T)
        st.subheader("Estilo")
        st.write(data["estilo"])
        st.subheader("Costos")
        st.write(data["costos"])

