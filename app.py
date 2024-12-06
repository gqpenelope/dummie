import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolios", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #1D1E2C;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: white !important;
    }
    .stApp {
        background-color: #1D1E2C;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Análisis y Optimización de Portafolios")

# Definición de ETFs y ventanas de tiempo
etfs = ['LQD', 'EMB', 'SPY', 'EMXC', 'IAU']
descripciones_etfs = {
    "LQD": {
        "nombre": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "exposicion": "Bonos corporativos denominados en dólares de EE.UU. y con grado de inversión.",
        "indice": "iBoxx $ Liquid Investment Grade Index",
        "moneda": "USD",
        "principales": ["JPMorgan Chase & Co", "Bank of America Corp", "Morgan Stanley"],
        "paises": "Estados Unidos y empresas multinacionales",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.14%",
        "beta": "0.53",
        "duracion": "8.45 años"
    },
    "EMB": {
        "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "exposicion": "Bonos de gobierno denominados en USD emitidos por países de mercados emergentes.",
        "indice": "JPMorgan EMBI Global Core Index",
        "moneda": "USD",
        "principales": ["Turkey (Republic of)", "Saudi Arabia (Kingdom of)", "Brazil Federative Republic of"],
        "paises": "América Latina, Medio Oriente, África y Asia",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.39%",
        "beta": "0.57",
        "duracion": "7.04 años"
    },
    "SPY": {
        "nombre": "iShares Core S&P 500 ETF",
        "exposicion": "Empresas de alta capitalización en Estados Unidos.",
        "indice": "S&P 500 Index (USD)",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos",
        "estilo": "Mix(Growth/Value)",
        "costos": "Comisión de administración: 0.03%",
        "beta": "1.01",
        "duracion": "NA"
    },
    "EMXC": {
        "nombre": "iShares MSCI Emerging Markets ex China ETF",
        "exposicion": "Empresas de mercados emergentes excluyendo China.",
        "indice": "MSCI Emerging Markets ex China Index",
        "moneda": "USD",
        "principales": ["Samsung Electronics", "Taiwan Semiconductor", "Infosys"],
        "paises": "Corea del Sur, Taiwán, India, entre otros",
        "estilo": "Growth",
        "costos": "Comisión de administración: 0.25%",
        "beta": "",
        "duracion": ""
    },
    "IAU": {
        "nombre": "iShares Gold Trust",
        "exposicion": "Inversión en oro físico como cobertura inflacionaria.",
        "indice": "LBMA Gold Price",
        "moneda": "USD",
        "principales": ["NA"],
        "paises": "Global",
        "estilo": "Commodity",
        "costos": "Comisión de administración: 0.25%",
        "beta": "0.13",
        "duracion": "NA"
    }
}
ventanas = {
    "2010-2023": ("2010-01-01", "2023-12-31"),
    "2010-2020": ("2010-01-01", "2020-12-31"),
    "2021-2023": ("2021-01-01", "2023-12-31")
}
# Menú lateral para seleccionar ventana de tiempo
with st.sidebar:
    selected = option_menu(
        menu_title="Ventana",
        options=list(ventanas.keys()),
        icons=["calendar", "calendar-range", "calendar3"],
        menu_icon="gear",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1D1E2C"},
            "icon": {"color": "white", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
                "background-color": "#1D1E2C",
            },
            "nav-link-selected": {"background-color": "#C4F5FC", "color": "white"},
        },
    )

start_date, end_date = ventanas[selected]

# Tabs de la aplicación
st.markdown(
    """
    <style>
    div[data-baseweb="tab-highlight"] {
        background-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
tab1, tab2, tab3, tab4 = st.tabs([
    "Análisis de Activos Individuales",
    "Portafolios Óptimos",
    "Comparación de Portafolios",
    "Black-Litterman"
])
# Descargar datos de los ETFs
@st.cache_data
def obtener_datos(etfs, start_date, end_date):
    datos = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    return datos

# Descargar datos del tipo de cambio USD/MXN
@st.cache_data
def obtener_tipo_cambio(start_date, end_date):
    tipo_cambio = yf.download('USDMXN=X', start=start_date, end=end_date)['Adj Close']
    return tipo_cambio

# Obtener datos
datos = obtener_datos(etfs, start_date, end_date)
tipo_cambio = obtener_tipo_cambio(start_date, end_date)

# Calcular rendimientos diarios
rendimientos = datos.pct_change().dropna()
# Función para calcular VaR y CVaR
def var_cvar(returns, confianza=0.95):
    VaR = returns.quantile(1 - confianza)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

# Función para calcular métricas adicionales
def calcular_metricas(rendimientos, benchmark=None, rf_rate=0.02):
    # Métricas básicas
    media = rendimientos.mean() * 252  # Rendimiento anualizado
    volatilidad = rendimientos.std() * np.sqrt(252)  # Volatilidad anualizada
    sharpe = (media - rf_rate) / volatilidad  # Ratio Sharpe
    sesgo = rendimientos.skew()  # Sesgo de los rendimientos
    curtosis = rendimientos.kurt()  # Curtosis de los rendimientos
    VaR, CVaR = var_cvar(rendimientos)
    momentum = rendimientos[-252:].sum() if len(rendimientos) >= 252 else np.nan # Momentum: suma de rendimientos de los últimos 12 meses
    
    # Sortino Ratio
    rendimientos_negativos = rendimientos[rendimientos < 0]
    downside_deviation = np.sqrt((rendimientos_negativos ** 2).mean()) * np.sqrt(252)
    sortino_ratio = media / downside_deviation if downside_deviation > 0 else np.nan
    
    # Drawdown
    rendimiento_acumulado = (1 + rendimientos).cumprod()
    max_acumulado = rendimiento_acumulado.cummax()
    drawdown = (rendimiento_acumulado / max_acumulado - 1).min()

    return {
        "Media": media,
        "Volatilidad": volatilidad,
        "Sharpe": sharpe,
        "Sesgo": sesgo,
        "Curtosis": curtosis,
        "VaR": VaR,
        "CVaR": CVaR,
        "Sortino Ratio": sortino_ratio,
        "Drawdown": drawdown,
        "Momentum": momentum,
    }

# Calcular métricas para cada ETF con SPY como benchmark
benchmark = rendimientos['SPY']  # Usamos SPY como índice de referencia
metricas = {
    etf: calcular_metricas(rendimientos[etf], benchmark if etf != 'SPY' else None)
    for etf in etfs
}
metricas_df = pd.DataFrame(metricas).T  # Convertir a DataFrame para análisis tabular

# Función para crear el histograma con hover interactivo
def histog_distr(returns, var_95, cvar_95, title):
    # Crear el histograma base
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Distribución de rendimientos",
        marker_color="#4CAF50",
        opacity=0.75
    ))

    # Añadir línea del VaR
    fig.add_vline(
        x=var_95,
        line_width=3,
        line_dash="dash",
        line_color="red"
    )
    fig.add_annotation(
        x=var_95,
        y=0,
        text=f"VaR (95%): {var_95:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color="red")
    )

    # Añadir línea del CVaR
    fig.add_vline(
        x=cvar_95,
        line_width=3,
        line_dash="dash",
        line_color="blue"
    )
    fig.add_annotation(
        x=cvar_95,
        y=0,
        text=f"CVaR (95%): {cvar_95:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-40,
        font=dict(color="blue")
    )

    # Configuración del diseño
    fig.update_layout(
        title=title,
        xaxis_title="Rendimientos",
        yaxis_title="Frecuencia",
        hovermode="x unified",  # Hover interactivo para el eje X
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        font=dict(color='white')  # Texto en blanco
    )

    return fig

# Tab 1: Análisis de Activos Individuales
with tab1:
    st.markdown(
        """
        <div style="
            background-color: #C4F5FC;
            padding: 8px;
            border-radius: 20px;
            color: black;
            text-align: center;
        ">
            <h1 style="margin: 0; color: #black; font-size: 25px; ">Análisis de Activos Individuales</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Selección del ETF para análisis
    etf_seleccionado = st.selectbox("Selecciona un ETF para análisis:", options=etfs)

    if etf_seleccionado:
        if etf_seleccionado not in datos.columns or datos[etf_seleccionado].dropna().empty:
            st.error(f"No hay datos disponibles para {etf_seleccionado} en la ventana seleccionada.")
        else:
            # Dividir en dos columnas
            col1, col2 = st.columns([3, 2])  # Relación 3:2 entre columnas izquierda y derecha

            # Columna Izquierda
            with col1:
                st.subheader("Características del ETF")
                data = descripciones_etfs[etf_seleccionado]

                # Tabla de características
                tabla_caracteristicas = pd.DataFrame({
                    "Características": ["Nombre", "Exposición", "Índice", "Moneda", "Principales Contribuyentes", "Países", "Estilo", "Costos", "Beta", "Duración"],
                    "Detalles": [
                        data["nombre"],
                        data["exposicion"],
                        data["indice"],
                        data["moneda"],
                        ", ".join(data["principales"]),
                        data["paises"],
                        data["estilo"],
                        data["costos"],
                        data["beta"],
                        data["duracion"]
                    ]
                })

                # Convertir el DataFrame a HTML y renderizarlo
                tabla_html = tabla_caracteristicas.to_html(index=False, escape=False)
                st.markdown(
                    """
                    <style>
                    table {
                        color: white;
                        background-color: transparent;
                        border-collapse: collapse;
                        width: 100%;
                    }
                    th {
                        background-color: #2CA58D;
                        color: black;
                        font-weight: bold;
                        text-align: center;
                        vertical-align: middle;
                    }
                    td {
                        border: 1px solid white;
                        padding: 8px;
                        text-align: center;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(tabla_html, unsafe_allow_html=True)

                # Gráfica de precios normalizados
                st.subheader("Serie de Tiempo de Precios Normalizados")
                precios_normalizados = datos[etf_seleccionado] / datos[etf_seleccionado].iloc[0] * 100
                fig_precios = go.Figure(go.Scatter(
                    x=precios_normalizados.index,
                    y=precios_normalizados,
                    mode='lines',
                    name=etf_seleccionado,
                    line=dict(color='#F46197')
                ))

                fig_precios.update_layout(
                    title=dict(text="Precio Normalizado", font=dict(color='white')),
                    xaxis=dict(title="Fecha", titlefont=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(title="Precio Normalizado", titlefont=dict(color='white'), tickfont=dict(color='white')),
                    hovermode="x unified",
                    plot_bgcolor='#1D1E2C',
                    paper_bgcolor='#1D1E2C',
                    font=dict(color='white')
                )
                fig_precios.update_xaxes(showgrid=False)
                fig_precios.update_yaxes(showgrid=False)
                st.plotly_chart(fig_precios)

            # Columna Derecha
            with col2:
                st.subheader("Métricas Calculadas")

                # Métricas en boxes
                style_metric_cards(background_color="#1F2C56", border_left_color="#F46197")
                st.markdown(
                    """
                    <style>
                    .metric-box {
                        background-color: #1F2C56;
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        margin-bottom: 10px;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                metricas = calcular_metricas(rendimientos[etf_seleccionado])
                st.columns(3)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Media", value=f"{metricas['Media']:.2f}")
                with col2:
                    st.metric(label="Volatilidad", value=f"{metricas['Volatilidad']:.2f}")
                with col3:
                    st.metric(label="Sharpe", value=f"{metricas['Sharpe']:.2f}")
                
                col4, col5,col6 = st.columns(3) 
                with col4:
                    st.metric(label="Sesgo", value=f"{metricas['Sesgo']:.2f}")
                with col5:
                    st.metric(label="Curtosis", value=f"{metricas['Curtosis']:.2f}")
                with col6:
                   st.metric(label="Sortino Ratio", value=f"{metricas['Sortino Ratio']:.2f}")
                    
                col7, col8, col9, col10 = st.columns(4)
                with col7:
                    st.metric(label="VaR", value=f"{metricas['VaR']:.2f}")
                with col8:
                    st.metric(label="CVaR", value=f"{metricas['CVaR']:.2f}")
                with col9:
                    st.metric(label="Drawdown", value=f"{metricas['Drawdown']:.2f}")
                
                col10 = st.columns(1)
                with col10:
                    st.metric(label="Momentum", value=f"{metricas['Momentum']:.2f}")
                    

                # Histograma de rendimientos
                st.subheader("Histograma de Rendimientos con VaR y CVaR")
                var_95, cvar_95 = var_cvar(rendimientos[etf_seleccionado], confianza=0.95)
                histograma = histog_distr(rendimientos[etf_seleccionado], var_95, cvar_95, f"Distribución de rendimientos para {etf_seleccionado}")
                st.plotly_chart(histograma)

# Tab 2: Portafolios Óptimos
with tab2:
    st.header("Portafolios Óptimos")
    st.markdown(
    """
    <div style="
        background-color: #C4F5FC;
        padding: 8px;
        border-radius: 20px;
        color: black;
        text-align: center;
    ">
        <h1 style="margin: 0; color: #black; font-size: 25px; ">Portafolios Óptimos</h1>
    </div>
    """,
    unsafe_allow_html=True,
    )

    # Función para optimizar portafolios
    def optimizar_portafolio(rendimientos, objetivo="sharpe", rendimiento_objetivo=None, incluir_tipo_cambio=False):
        media = rendimientos.mean() * 252
        covarianza = rendimientos.cov() * 252
        num_activos = len(media)
        pesos_iniciales = np.ones(num_activos) / num_activos
        limites = [(0.035, 0.4) for _ in range(num_activos)]
        restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if incluir_tipo_cambio:
            tipo_cambio_rendimientos = tipo_cambio.pct_change().mean() * 252
            media += tipo_cambio_rendimientos

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

    # Optimización de los portafolios
    pesos_sharpe = optimizar_portafolio(rendimientos, objetivo="sharpe")
    pesos_volatilidad = optimizar_portafolio(rendimientos, objetivo="volatilidad")
    pesos_rendimiento = optimizar_portafolio(rendimientos, objetivo="rendimiento", rendimiento_objetivo=0.10, incluir_tipo_cambio=True)

    # Mostrar los pesos optimizados
    pesos_df = pd.DataFrame({
        "Máximo Sharpe": pesos_sharpe,
        "Mínima Volatilidad": pesos_volatilidad,
        "Mínima Volatilidad (Rendimiento 10% en MXN)": pesos_rendimiento
    }, index=etfs)

    # Visualización de portafolios optimizados
    portafolio_seleccionado = st.selectbox(
        "Selecciona el portafolio a visualizar:",
        ["Máximo Sharpe", "Mínima Volatilidad", "Mínima Volatilidad (Rendimiento 10% en MXN)"]
    )

    st.subheader(f"Pesos del Portafolio: {portafolio_seleccionado}")

    fig_barras = go.Figure(data=[
        go.Bar(
            x=pesos_df.index,
            y=pesos_df[portafolio_seleccionado],
            marker_color='#2CA58D'
        )
    ])

    fig_barras.update_layout(
        title=dict(
            text=f"Pesos del Portafolio: {portafolio_seleccionado}",
            font=dict(color='white')
        ),
        xaxis=dict(
            title="ETFs",
            titlefont=dict(color='white'),
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title="Pesos",
            titlefont=dict(color='white'),
            tickfont=dict(color='white')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig_barras)

    # Gráfica de pastel
    st.subheader("Composición del Portafolio")

    valores_redondeados = [round(peso, 6) if peso > 1e-6 else 0 for peso in pesos_df[portafolio_seleccionado]]
    etiquetas = [
        f"{etf} ({peso:.6f})" if peso > 0 else f"{etf} (<1e-6)"
        for etf, peso in zip(etfs, pesos_df[portafolio_seleccionado])
    ]

    fig_pastel = go.Figure(data=[
        go.Pie(
            labels=etiquetas,
            values=valores_redondeados,
            hoverinfo='label+percent+value',
            textinfo='percent',
            marker=dict(colors=['#2CA58D', '#F46197', '#84BC9C', '#FFD700', '#497076'])
        )
    ])

    fig_pastel.update_layout(
        title=dict(
            text=f"Distribución del Portafolio ({portafolio_seleccionado})",
            font=dict(color='white')
        ),
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig_pastel)

    if portafolio_seleccionado == "Mínima Volatilidad (Rendimiento 10% en MXN)":
        st.subheader("Ajustes por Tipo de Cambio")
        try:
            tipo_cambio_medio = tipo_cambio.mean()
            if isinstance(tipo_cambio_medio, pd.Series):
                tipo_cambio_medio = tipo_cambio_medio.iloc[0]
            st.write(f"**Tipo de cambio medio esperado:** {tipo_cambio_medio:.2f} USD/MXN")
        except Exception as e:
            st.error(f"Error al calcular el promedio del tipo de cambio: {e}")

# Tab 3: Comparación de Portafolios
with tab3:
    st.header("Comparación de Portafolios")

    # Comparación de Precios Normalizados
    st.subheader("Precios Normalizados de los ETFs Seleccionados")
    precios_normalizados = datos / datos.iloc[0] * 100

    fig = go.Figure()
    for etf in etfs:
        fig.add_trace(go.Scatter(
            x=precios_normalizados.index,
            y=precios_normalizados[etf],
            mode='lines',
            name=etf
        ))

    fig.update_layout(
        title="Comparación de Precios Normalizados",
        xaxis_title="Fecha",
        yaxis_title="Precio Normalizado",
        hovermode="x unified",
        plot_bgcolor='#1D1E2C',
        paper_bgcolor='#1D1E2C',
        font=dict(color='white')
    )
    st.plotly_chart(fig)

    # Backtesting
    st.subheader("Backtesting")

    def backtesting_portafolio(rendimientos, pesos, inicio, fin, nivel_var=0.05):
        rendimientos_bt = rendimientos.loc[inicio:fin]
        rendimientos_portafolio = rendimientos_bt.dot(pesos)
        rendimiento_acumulado = (1 + rendimientos_portafolio).cumprod()

        # Estadísticas básicas
        rendimiento_anualizado = (1 + rendimientos_portafolio.mean()) ** 252 - 1
        volatilidad_anualizada = rendimientos_portafolio.std() * np.sqrt(252)
        sharpe_ratio = rendimiento_anualizado / volatilidad_anualizada
        sesgo_portafolio = rendimientos_portafolio.skew()
        curtosis_portafolio = rendimientos_portafolio.kurt()

        # Cálculo de VaR y CVaR
        var = np.percentile(rendimientos_portafolio, nivel_var * 100)
        cvar = rendimientos_portafolio[rendimientos_portafolio <= var].mean()

        # Sortino Ratio
        rendimientos_negativos = rendimientos_portafolio[rendimientos_portafolio < 0]
        downside_deviation = np.sqrt((rendimientos_negativos ** 2).mean()) * np.sqrt(252)
        sortino_ratio = rendimiento_anualizado / downside_deviation

        # Drawdown
        max_acumulado = rendimiento_acumulado.cummax()
        drawdown = (rendimiento_acumulado / max_acumulado - 1).min()

        # Diccionario con todas las estadísticas
        estadisticas = {
            "Rendimiento Anualizado": rendimiento_anualizado,
            "Volatilidad Anualizada": volatilidad_anualizada,
            "Ratio de Sharpe": sharpe_ratio,
            "Sesgo": sesgo_portafolio,
            "Curtosis": curtosis_portafolio,
            "VaR ({}%)".format(int(nivel_var * 100)): var,
            "CVaR ({}%)".format(int(nivel_var * 100)): cvar,
            "Sortino Ratio": sortino_ratio,
            "Máximo Drawdown": drawdown
        }

        return rendimiento_acumulado, estadisticas

    # Parámetros del backtesting
    inicio = "2021-01-01"
    fin = "2023-12-31"

    # Pesos para los diferentes portafolios
    pesos_iguales = np.full(rendimientos.shape[1], 1 / rendimientos.shape[1])
    bt_sharpe, stats_sharpe = backtesting_portafolio(rendimientos, pesos_sharpe, inicio, fin)
    bt_volatilidad, stats_volatilidad = backtesting_portafolio(rendimientos, pesos_volatilidad, inicio, fin)
    bt_rendimiento, stats_rendimiento = backtesting_portafolio(rendimientos, pesos_rendimiento, inicio, fin)
    bt_iguales, stats_iguales = backtesting_portafolio(rendimientos, pesos_iguales, inicio, fin)

    # Gráfica de rendimiento acumulado
    st.subheader("Rendimiento Acumulado")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_sharpe.index, y=bt_sharpe, mode='lines', name="Máximo Sharpe"))
    fig_bt.add_trace(go.Scatter(x=bt_volatilidad.index, y=bt_volatilidad, mode='lines', name="Mínima Volatilidad"))
    fig_bt.add_trace(go.Scatter(x=bt_rendimiento.index, y=bt_rendimiento, mode='lines', name="Mínima Volatilidad (Rendimiento 10%)"))
    fig_bt.add_trace(go.Scatter(x=bt_iguales.index, y=bt_iguales, mode='lines', name="Pesos Iguales"))

    fig_bt.update_layout(
        title="Rendimiento Acumulado",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Acumulado",
        hovermode="x unified",
        plot_bgcolor='#1D1E2C',
        paper_bgcolor='#1D1E2C',
        font=dict(color='white')
    )
    st.plotly_chart(fig_bt)

    # Mostrar estadísticas
    st.markdown("### Métricas de Backtesting")
    for nombre, stats in [("Máximo Sharpe", stats_sharpe), ("Mínima Volatilidad", stats_volatilidad), 
                          ("Mínima Volatilidad (Rendimiento 10%)", stats_rendimiento), ("Pesos Iguales", stats_iguales)]:
        st.markdown(f"**{nombre}:**")
        for key, value in stats.items():
            st.metric(label=key, value=f"{value:.2f}")

# Tab 4: Black-Litterman
with tab4:
    st.header("Modelo de Optimización Black-Litterman")

    try:
        # Parámetros del modelo
        media_rendimientos = rendimientos.mean() * 252
        covarianza_rendimientos = rendimientos.cov() * 252
        P = np.eye(len(etfs))  # Views: una opinión por ETF
        Q = np.array([0.08, 0.065, 0.12, 0.06, 0.05])  # Opiniones esperadas (rendimientos)

        def black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q, tau=0.05):
            pi = media_rendimientos
            omega = np.diag(np.diag(P @ covarianza_rendimientos @ P.T)) * tau

            # Cálculo de medias ajustadas
            medio_ajustado = np.linalg.inv(
                np.linalg.inv(tau * covarianza_rendimientos) + P.T @ np.linalg.inv(omega) @ P
            ) @ (
                np.linalg.inv(tau * covarianza_rendimientos) @ pi + P.T @ np.linalg.inv(omega) @ Q
            )

            restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Pesos deben sumar 1

            def objetivo_func(pesos):
                return -np.dot(pesos, medio_ajustado) / np.sqrt(np.dot(pesos.T, np.dot(covarianza_rendimientos, pesos)))

            resultado = sco.minimize(objetivo_func, np.ones(len(media_rendimientos)) / len(media_rendimientos),
                                     method='SLSQP', bounds=[(0.035, 0.4) for _ in range(len(media_rendimientos))],
                                     constraints=restricciones)
            return resultado.x

        # Optimizar el portafolio
        pesos_black_litterman = black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q)

        # Expectativas de crecimiento para cada ETF
        st.subheader("Expectativas de Crecimiento por ETF")
        expectativas = {
            "LQD": "Esperamos buen crecimiento debido a su alta duración y exposición a sectores clave como banca y tecnología. Proyección: 8%.",
            "EMB": "Beneficio por bajas tasas de interés en mercados emergentes y exposición a bonos gubernamentales mexicanos. Proyección: 6.5%.",
            "SPY": "Exposición significativa a tecnología (31.66%) y políticas proteccionistas en EE.UU. Proyección: 12%.",
            "EMXC": "Diversificación en mercados emergentes excluyendo China, con alta exposición a tecnología. Proyección: 6%.",
            "IAU": "Commodities como cobertura inflacionaria; alta demanda de bancos centrales. Proyección: 5%."
        }

        for etf, expectativa in expectativas.items():
            st.write(f"**{etf}:** {expectativa}")

        # Mostrar restricciones del portafolio
        st.subheader("Restricciones del Portafolio")
        st.write("- Peso mínimo por activo: 3.5%")
        st.write("- Peso máximo por activo: 40%")

        # Mostrar los pesos optimizados
        st.subheader("Pesos Ajustados del Portafolio Black-Litterman")
        for etf, peso in zip(etfs, pesos_black_litterman):
            st.write(f"**{etf}:** {peso:.2%}")

        # Gráfica de pastel
        fig_bl = go.Figure(data=[
            go.Pie(labels=etfs, values=pesos_black_litterman, hoverinfo='label+percent')
        ])
        fig_bl.update_layout(
            title="Distribución del Portafolio Ajustado - Black-Litterman",
            legend=dict(font=dict(color="white")),
            paper_bgcolor='#1D1E2C',
            font=dict(color='white')
        )
        st.plotly_chart(fig_bl)
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

