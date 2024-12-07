# Importamos las librerias pandas y numpy
from curses.ascii import alt
import pandas as pd
import numpy as np
import scipy.stats as stats

# Importamos la libreria yahoo finance
import yfinance as yf

# Importamos cufflinks
import cufflinks as cf
cf.set_config_file(offline=True)
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from numpy.linalg import multi_dot
import scipy.optimize as sco
import Funciones as fun
#import seaborn as sns


#
import streamlit as st
st.set_page_config(page_title="Proyecto", layout="wide")

tab1, tab2 = st.tabs(["Ejercicio 1", "Ejercicio 2"])
with tab1: 
    tab3, tab4, tab5, tab6 = st.tabs(["Inciso a)", "Incisco b)", "Inciso c)", "Inciso d)"])
    with tab3: 
        st.write("A continuación, se presentan cinco fondos cotizados en bolsa (ETF’s) seleccionados con el objetivo de diversificar y optimizar una cartera de inversión. Estos fondos están denominados en la misma divisa, lo cual facilita su comparación directa. En nuestra selección se ha incluido un ETF de cada una de las siguientes categorías: renta fija desarrollada, renta fija emergente, renta variable desarrollada, renta variable emergente, y un ETF adicional que cubre materias primas o crédito. Cada uno de estos activos ofrece una exposición a diferentes mercados y sectores, permitiendo así un balance adecuado entre riesgo y rentabilidad.")

        
        activos = [
            "TLT - iShares 20+ Year Treasury Bond ETF",
            "EMB - iShares J.P. Morgan USD Emerging Markets Bond ETF",
            "SPY - SPDR S&P 500 ETF Trust",
            "EEM - iShares MSCI Emerging Markets ETF",
            "GLD - SPDR Gold Shares ETF"
            ]
        # Información adicional de cada activo
        informacion_activos = {
            "TLT - iShares 20+ Year Treasury Bond ETF": """

                **Objetivo:** Seguir el rendimiento del ICE U.S. Treasury 20+ Year Bond Index, que incluye bonos del Tesoro de EE. UU. con vencimientos de 20 años o más.

                **Exposición:** Bonos del Tesoro de EE. UU. a largo plazo (20 años o más).

                **Moneda de denominación:** Dólares estadounidenses (USD).

                **Índice de referencia:** ICE U.S. Treasury 20+ Year Bond Index.

                **Duración promedio:** Alta (más de 15 años), lo que lo hace sensible a los cambios en las tasas de interés.

                **Riesgo:** Riesgo principalmente por cambios en tasas de interés; bajo riesgo de crédito, ya que invierte en bonos del gobierno de EE. UU.

                **Costos:** Ratio de gastos de 0.15%.

                **Liquidez:** Alta, con alto volumen de negociación.

                **Dividendos:** Distribuye pagos mensuales de intereses generados por los bonos.

                **Estilo de inversión:** Renta fija de grado de inversión.
            """,
            "EMB - iShares J.P. Morgan USD Emerging Markets Bond ETF": """
                **Objetivo:** Seguir el rendimiento del J.P. Morgan EMBI Global Core Index, que incluye bonos soberanos emitidos por países emergentes denominados en dólares estadounidenses.
                
                **Exposición:** Bonos soberanos de mercados emergentes denominados en dólares (países como Brasil, México, Turquía, entre otros).
                
                **Moneda de denominación:** Dólares estadounidenses (USD).
                
                **Índice de referencia:** J.P. Morgan EMBI Global Core Index.
                
                **Duración promedio:** Alta (más de 15 años), lo que lo hace sensible a los cambios en las tasas de interés.
                
                **Riesgo:**  Exposición a riesgo de crédito y riesgo país, ya que invierte en mercados emergentes con niveles variables de estabilidad económica y política.
                
                **Costos:** Ratio de gastos de aproximadamente 0.39%
                
                **Liquidez:** Moderada a alta, pero menor que otros ETFs de bonos del gobierno de EE. UU.
                
                **Dividendos:** Distribuye pagos mensuales de intereses generados por los bonos.
                
                **Estilo de inversión:** Renta fija de mercados emergentes, con una combinación de riesgo y rendimiento más alto comparado con los bonos de mercados desarrollados.
            """,
            "SPY - SPDR S&P 500 ETF Trust": """
                **Objetivo:** Seguir el rendimiento del S&P 500 Index, que incluye las 500 empresas más grandes y representativas de EE. UU.    
                
                **Exposición:** Renta variable (acciones de grandes empresas estadounidenses) de diversos sectores como tecnología, salud, consumo, energía, etc. 
                
                **Moneda de denominación:** Dólares estadounidenses (USD).
                
                **Índice de referencia:** S&P 500 Index.
                
                **Duración promedio:** No aplica.
                
                **Riesgo:** Exposición a riesgo de mercado, dado que el valor de las acciones está sujeto a la volatilidad del mercado y condiciones económicas.
                
                **Costos:** Ratio de gastos de aproximadamente 0.09%, lo que lo hace uno de los ETFs más eficientes en costos.
                
                **Liquidez:** Alta, siendo uno de los ETFs más grandes y negociados en el mercado global.
                
                **Dividendos: Distribuye pagos trimestrales de dividendos generados por las empresas del índice.
                
                **Estilo de Inversión**: Acciones de gran capitalización con enfoque en valor y crecimiento (growth & value).
            """,
            "EEM - iShares MSCI Emerging Markets ETF": """

                **Objetivo:** Seguir el rendimiento del MSCI Emerging Markets Index, que incluye acciones de empresas de mercados emergentes de todo el mundo, como China, India, Brasil, Sudáfrica, entre otros.   
                
                **Exposición:** Renta variable de mercados emergentes, con acciones de empresas de sectores como tecnología, energía, materiales, salud, etc.
                
                **Moneda de denominación:** Dólares estadounidenses (USD).
                
                **Índice de referencia:** MSCI Emerging Markets Index.
                
                **Duración promedio:** No aplica.
                
                **Riesgo:** Exposición a riesgo de mercado, riesgo país y riesgo de divisas, dado que invierte en mercados con economías menos estables y monedas volátiles.
                
                **Costos:** Ratio de gastos de aproximadamente 0.68%.
                
                **Liquidez:** Alta, aunque menor que ETFs más grandes como el SPY, sigue siendo muy negociado.
                
                **Dividendos: Distribuye pagos trimestrales de dividendos generados por las acciones de empresas en los mercados emergentes.
                
                **Estilos de Inversión*: Renta variable en mercados emergentes, con una mezcla de crecimiento (growth) y valor (value), dependiendo del país y sector.
            """,
            "GLD - SPDR Gold Shares ETF": """
                **Objetivo:** Seguir el rendimiento del precio del oro, mediante la inversión en lingotes de oro físico.
                
                **Exposición:** Oro físico, con el ETF respaldado por lingotes de oro almacenados en custodia.
                
                **Moneda de denominación:** Dólares estadounidenses (USD).
                
                **Índice de referencia:** No sigue un índice específico, sino que busca reflejar el precio del oro en los mercados internacionales.
                
                **Duración promedio:** No aplica.
                
                **Riesgo:** Exposición al riesgo de precios del oro, que pueden verse influenciados por factores como la oferta y demanda, inflación, tasas de interés, y la incertidumbre económica o política global.
                
                **Costos:** Ratio de gastos de aproximadamente 0.40%.
                
                **Liquidez:** Alta, siendo uno de los ETFs de metales preciosos más grandes y negociados.
                
                **Dividendos: No distribuye dividendos, ya que invierte en oro físico.
                
                **Estilos de Inversión*: Activos refugio (commodities), ideal para diversificación en carteras como cobertura contra la inflación o crisis económicas.
            """
        }

        # Inicializar estado para los botones si no existe
        for activo in activos:
            if f"mostrar_info_{activo}" not in st.session_state:
                st.session_state[f"mostrar_info_{activo}"] = False

        # Función para alternar el estado de un botón específico
        def toggle_info(activo):
            st.session_state[f"mostrar_info_{activo}"] = not st.session_state[f"mostrar_info_{activo}"]
            
        # Título de la aplicación
        st.title("Mi selección de Activos")

        # Aplicar estilo CSS para hacer que todos los botones tengan el mismo tamaño
        st.markdown("""
                <style>
                    .stButton button {
                        width: 100%;  /* Ajusta el ancho a 100% del contenedor */
                        height: 50px; /* Ajusta la altura de los botones */
                    }
                </style>
        """, unsafe_allow_html=True)
        
        # Crear botones para cada activo
        for activo in activos:
            if st.button(f"{activo}", key=activo, on_click=toggle_info, args=(activo,)):
                pass
            
            # Mostrar u ocultar la información del activo según el estado
            if st.session_state[f"mostrar_info_{activo}"]:
                st.write(informacion_activos[activo])


    with tab4: 
        etfs = ["TLT", "EMB", "SPY", "EEM", "GLD"]
        p_cierre = fun.obtener_datos_acciones(etfs, "2010-01-01", "2023-12-31")
        rendimientos = fun.calcular_rendimientos(p_cierre)
        st.dataframe(rendimientos)

        # Agregar un selector para elegir el rango de fechas
        start_date = st.date_input("Fecha de inicio", min_value=pd.to_datetime('2010-01-01'), value=pd.to_datetime('2010-01-01'))
        end_date = st.date_input("Fecha de fin", min_value=pd.to_datetime('2020-12-31'), value=pd.to_datetime('2020-12-31'))
        
        # Filtrar los datos con las fechas seleccionadas
        retornos_filtrados = rendimientos.loc[start_date:end_date]
        
        colores = {
            "TLT": "red",
            "EMB": "blue",
            "SPY": "green",
            "EEM": "orange",
            "GLD": "purple"}

        # Graficar solo los datos filtrados
        for col in retornos_filtrados.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(retornos_filtrados.index, retornos_filtrados[col], marker='o', linestyle='-', color=colores[col])
            ax.set_title(f"Serie de tiempo de retornos diarios: {col}")
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Retornos diarios')
            ax.tick_params(axis='x', rotation=45)
            
            st.pyplot(fig)


       
        medias = [fun.metricas(etf)[0] for etf in etfs]
        sesgos = [fun.metricas(etf)[1] for etf in etfs]
        kurtosis = [fun.metricas(etf)[2] for etf in etfs]
        var = [fun.metricas(etf)[3] for etf in etfs]
        cvar = [fun.metricas(etf)[4] for etf in etfs]
        sharpe = [fun.metricas(etf)[5] for etf in etfs]
        sortino = [fun.metricas(etf)[6] for etf in etfs]
        drawdown = [fun.metricas(etf)[7] for etf in etfs]

        etfs = ["TLT", "EMB", "SPY", "EEM", "GLD"]
        metricas = ["Media", "Sesgo", "Kurtosis", "Var", "CVar", "Sharpe", "Sortino", "Drawdown"]

        # Crear DataFrame con las métricas como columnas y ETFs como índice
        df_1 = pd.DataFrame({
           "Media": medias,
           "Sesgo": sesgos,
           "Kurtosis": kurtosis,
           "Var": var,
           "CVar": cvar,
           "Sharpe": sharpe,
           "Sortino": sortino,
           "Drawdown": drawdown
        }, index=etfs)

        # Transponer df_1 para obtener las métricas como filas
        df_1_t = df_1.T.reset_index()
        df_1_t.columns = ["Métrica"] + etfs  # Renombrar columnas
        
        # Mostrar en Streamlit (sin columna adicional)
        st.dataframe(df_1_t)


        





    with tab5:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from scipy.optimize import minimize
        import matplotlib.pyplot as plt
        import streamlit as st
        
        # Configuración
        etfs = ['TLT', 'EMB', 'SPY', 'EEM', 'GLD']
        start_date = '2010-01-01'
        end_date = '2020-12-31'
        
        # Descargar precios ajustados
        prices = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
        
        # Calcular rendimientos diarios
        returns = prices.pct_change().dropna()
        
        # Calcular la matriz de covarianzas
        cov_matrix = returns.cov()
        
        # Función de optimización: minimizar la volatilidad
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Restricciones: suma de pesos = 1, pesos >= 0
        constraints = ({
            'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1
            })
        
        bounds = [(0, 1) for _ in range(len(etfs))]  # No se permite apalancamiento
        
        # Puntos iniciales: asignación igualitaria
        initial_weights = np.ones(len(etfs)) / len(etfs)
        
        # Optimización
        result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Pesos óptimos
        optimal_weights = result.x
        
        # Resultados en Streamlit
        st.title("Optimización del Portafolio de Mínima Volatilidad")
        st.subheader("Pesos óptimos del portafolio de mínima volatilidad:")
        
        # Crear un DataFrame para mostrar los pesos óptimos
        weights_df = pd.DataFrame({
            'ETF': etfs,
            'Peso óptimo': optimal_weights
            })
        
        # Mostrar la tabla con los resultados
        st.dataframe(weights_df)  # Mostrar los pesos en una tabla
        
        # Calcular y mostrar la volatilidad del portafolio
        min_vol = portfolio_volatility(optimal_weights, cov_matrix)
        st.subheader(f"\nVolatilidad mínima del portafolio: {min_vol:.4%}")
        
        # Graficar los pesos óptimos en un gráfico de barras
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(etfs, optimal_weights, color='skyblue')
        ax.set_title("Pesos óptimos del portafolio de mínima volatilidad")
        ax.set_ylabel("Pesos")
        st.pyplot(fig)  # Mostrar el gráfico en Streamlit
        




        
        import scipy.optimize as sco
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import matplotlib.pyplot as plt
        
        # Descargar precios de los activos
        symbols = ['TLT', 'EMB', 'SPY', 'EEM', 'GLD']
        start_date = '2010-01-01'
        end_date = '2020-12-31'
        
        # Descargar datos de precios ajustados desde Yahoo Finance
        @st.cache_data
        def get_prices():
            prices = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
            retornos = prices.pct_change().dropna()
            return prices, retornos
        
        prices, retornos = get_prices()
        # Función para calcular estadísticas del portafolio
        def portfolio_stats(weights):
            weights = np.array(weights)[:, np.newaxis]
            # Cálculo de retornos anuales
            port_rets = weights.T @ np.array(retornos.mean() * 252)[:, np.newaxis]
            # Cálculo de volatilidad anual
            port_vols = np.sqrt(np.linalg.multi_dot([weights.T, retornos.cov() * 252, weights]))
            # Cálculo de Sharpe Ratio
            sharpe_ratio = port_rets / port_vols
            return port_rets[0, 0], port_vols[0, 0], sharpe_ratio[0, 0]
        
        # Función objetivo para minimizar el Sharpe ratio negativo (maximizar el Sharpe)
        def min_sharpe_ratio(weights):
            return -portfolio_stats(weights)[2]
            
        # Configuración de la optimización
        numdeact = len(symbols)
        initial_wts = np.array([1 / numdeact] * numdeact)  # Pesos iniciales iguales
        bnds = [(0, 1)] * numdeact  # Límites de los pesos
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Restricción de que la suma de los pesos sea 1
        
        # Optimización del Sharpe Ratio
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
        
        # Mostrar resultados en Streamlit
        st.title("Optimización de Portafolio: Maximizar el Sharpe Ratio")
        
        # Pesos óptimos y resultados
        st.subheader("Pesos óptimos del portafolio")
        optimal_weights = opt_sharpe['x']
        optimal_weights_rounded = [round(w * 100, 2) for w in optimal_weights]
        weights_df = pd.DataFrame({'Activo': symbols, 'Porcentaje (%)': optimal_weights_rounded})
        
        # Mostrar tabla con los pesos en Streamlit
        st.table(weights_df)
        
        # Estadísticas del portafolio
        stats = ['Retornos Anuales', 'Volatilidad Anual', 'Sharpe Ratio']
        portfolio_values = portfolio_stats(opt_sharpe['x'])
        portfolio_values_rounded = [round(value, 4) for value in portfolio_values]
        stats_df = pd.DataFrame({'Métrica': stats, 'Valor': portfolio_values_rounded})
        st.subheader("Estadísticas del portafolio")
        st.table(stats_df)
        
        # Gráfica circular para visualizar la distribución de pesos
        st.subheader("Distribución de Pesos del Portafolio")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(optimal_weights_rounded, labels=symbols, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20c(np.arange(len(symbols)) / len(symbols)))
        ax.axis('equal')  # Asegura que la gráfica sea circular
        st.pyplot(fig)
        
        # Graficar la evolución de los precios (opcional)
        st.subheader("Evolución de los Precios de los Activos")
        fig, ax = plt.subplots(figsize=(10, 6))
        for symbol in symbols:
            ax.plot(prices.index, prices[symbol], label=symbol)
        ax.set_title("Evolución de los Precios Ajustados")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio Ajustado")
        ax.legend(loc='best')
        st.pyplot(fig)









        import numpy as np
        import pandas as pd
        import yfinance as yf
        from scipy.optimize import minimize
        import matplotlib.pyplot as plt
        import streamlit as st
        
        # Configuración de Streamlit
        st.title("Optimización de Portafolio con Rendimiento Objetivo")
        st.sidebar.header("Parámetros")
        start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime('2010-01-01'))
        end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime('2020-12-31'))
        target_return = st.sidebar.slider("Rendimiento objetivo (anualizado)", 0.05, 0.20, 0.10, step=0.01)
        
        # Descargar precios ajustados de los ETFs
        etfs = ['TLT', 'EMB', 'SPY', 'EEM', 'GLD']
        prices = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
        
        # Descargar el tipo de cambio USD/MXN
        usd_mxn = yf.download('MXN=X', start=start_date, end=end_date)['Adj Close']
        
        # Asegurar que las fechas coincidan entre prices y usd_mxn
        prices, usd_mxn = prices.align(usd_mxn, join='inner', axis=0)
        usd_mxn = usd_mxn.squeeze()
        
        # Convertir precios a pesos mexicanos
        prices_mxn = prices.multiply(usd_mxn, axis=0)
        
        # Eliminar valores faltantes
        prices_mxn.dropna(inplace=True)
        
        # Calcular rendimientos diarios en pesos mexicanos
        returns = prices_mxn.pct_change().dropna()
        
        # Calcular el rendimiento anualizado y matriz de covarianzas
        mean_returns = returns.mean() * 252  # Rendimiento esperado anualizado
        cov_matrix = returns.cov()
        
        # Función de optimización: minimizar la volatilidad
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Restricciones
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Pesos suman 1
            {'type': 'ineq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return},  # Rendimiento objetivo
            )
        bounds = [(0, 1) for _ in range(len(etfs))]  # Pesos entre 0 y 1
        
        # Puntos iniciales: asignación igualitaria
        initial_weights = np.ones(len(etfs)) / len(etfs)
        
        # Optimización
        result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Pesos óptimos
        optimal_weights = result.x
        
        # Calcular métricas del portafolio
        min_vol = portfolio_volatility(optimal_weights, cov_matrix)
        portfolio_return = np.dot(optimal_weights, mean_returns)
        
        # Mostrar resultados en Streamlit
        st.header("Resultados del Portafolio Optimizado")
        st.subheader("Pesos Óptimos")
        weights_df = pd.DataFrame({
            'ETF': etfs,
            'Peso Óptimo': optimal_weights
            })
        st.dataframe(weights_df)
        
        st.subheader("Estadísticas del Portafolio")
        stats_df = pd.DataFrame({
            "Métrica": ["Rendimiento Esperado", "Volatilidad Mínima"],
            "Valor": [f"{portfolio_return:.2%}", f"{min_vol:.2%}"]
            })
        st.dataframe(stats_df)
        
        # Graficar pesos óptimos
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(etfs, optimal_weights, color='skyblue')
        ax.set_title("Pesos Óptimos del Portafolio")
        ax.set_ylabel("Peso")
        st.pyplot(fig)


    with tab6:
        # Import base libraries
        import streamlit as st
        import numpy as np
        from numpy.linalg import multi_dot
        import pandas as pd
        import yfinance as yf

        # Ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        # Configuración de Streamlit
        st.title("Análisis de Portafolio")
        st.write("Este dashboard analiza el rendimiento y métricas de un portafolio y sus componentes.")

        # Parámetros iniciales
        max_shape = [0.0, 0.0, .1749, .5537, .2714]
        df_back = yf.download(['TLT', 'EMB', 'SPY', 'EEM', 'GDL'], start='2021-01-01', end='2023-12-31', progress=False)['Close']

        # Mostrar datos descargados
        st.subheader("Datos descargados")
        st.table(df_back)

        # Calcular retornos
        retornos_back = df_back.pct_change().fillna(0)

        st.subheader("Retornos diarios")
        st.table(retornos_back)

        # Calcular el valor del portafolio
        precios = np.array(df_back)
        w = np.array(max_shape)
        valor_portafolio = precios @ w.T

        retornos_back["ACWI"] = valor_portafolio

        abc = retornos_back["ACWI"].to_frame()
        abc = abc.rename(columns={"ACWI": "Valor"})

        st.subheader("Valor del portafolio")
        st.table(abc)

        # Calcular rendimientos anuales
        a = abc.iloc[0]
        b = abc.iloc[252]
        c = abc.iloc[503]
        d = abc.iloc[-1]

        rendimientos_A = b / a - 1
        rendimientos_B = c / b - 1
        rendimientos_C = d / c - 1
        rendimientos_acumulados = (1 + rendimientos_A) * (1 + rendimientos_B) * (1 + rendimientos_C) - 1

        st.subheader("Rendimientos anuales y acumulados")
        st.table(pd.DataFrame({
            "2021": [rendimientos_A],
            "2022": [rendimientos_B],
            "2023": [rendimientos_C],
            "Acumulado": [rendimientos_acumulados]
        }))

        # Calcular métricas para el portafolio completo
        media = np.mean(retornos_back["ACWI"])
        sesgo = retornos_back["ACWI"].skew()
        kurt = retornos_back["ACWI"].kurtosis()
        VaR_95 = retornos_back["ACWI"].quantile(0.05)
        CVaR_95 = retornos_back["ACWI"][retornos_back["ACWI"] <= VaR_95].mean()

        st.subheader("Métricas del portafolio")
        st.table(pd.DataFrame({
            "Media": [media],
            "Sesgo": [sesgo],
            "Curtosis": [kurt],
            "VaR 95%": [VaR_95],
            "CVaR 95%": [CVaR_95]
        }))

        # Funciones para métricas avanzadas
        def sharpe_ratio(retornos, risk_free_rate=0.02):
            mean_return = retornos.mean()
            std_dev = retornos.std()
            return (mean_return - risk_free_rate / 252) / std_dev

        def sortino_ratio(retornos, risk_free_rate=0.02):
            mean_return = retornos.mean()
            downside_std = retornos[retornos < 0].std()
            return (mean_return - risk_free_rate / 252) / downside_std

        def max_drawdown(prices):
            cumulative = (1 + prices).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            return drawdown.min()

        # Calcular métricas para cada ETF
        metricas = {}
        for etf in retornos_back.columns:
            metricas[etf] = {
                'Sharpe Ratio': sharpe_ratio(retornos_back[etf]),
                'Sortino Ratio': sortino_ratio(retornos_back[etf]),
                'Max Drawdown': max_drawdown(retornos_back[etf]),
            }

        metricas_df = pd.DataFrame(metricas).T
        st.subheader("Métricas por ETF")
        st.table(metricas_df)

        # Información del S&P500
        df_sp500 = yf.download(['^GSPC'], start='2021-01-01', end='2023-12-31', progress=False)['Close']
        retornos_sp500 = df_sp500.pct_change().fillna(0)

        # Calcular métricas para el S&P500
        metricas_sp500 = {}
        for etf in df_sp500.columns:
            metricas_sp500[etf] = {
                'Sharpe Ratio': sharpe_ratio(retornos_sp500[etf]),
                'Sortino Ratio': sortino_ratio(retornos_sp500[etf]),
                'Max Drawdown': max_drawdown(retornos_sp500[etf]),
            }

        metricas_df_sp500 = pd.DataFrame(metricas_sp500).T
        st.subheader("Métricas S&P500")
        st.table(metricas_df_sp500)

        # Portafolio Equitativo
        w_equitativo = np.array([.2, .2, .2, .2, .2])
        valor_portafolio_equitativo = precios @ w_equitativo.T

        aaa = pd.DataFrame({'Equitativo': valor_portafolio_equitativo})
        ret_equitativo = aaa.pct_change().fillna(0)

        media_equitativo = np.mean(ret_equitativo)
        sesgo_equitativo = ret_equitativo.skew()
        kurt_equitativo = ret_equitativo.kurtosis()

        st.subheader("Métricas Equitativas")
        st.table(pd.DataFrame({
            "Media": [media_equitativo],
            "Sesgo": [sesgo_equitativo],
            "Curtosis": [kurt_equitativo],
        }))

        # Calcular métricas para el portafolio equitativo
        VaR_95_equi = ret_equitativo.quantile(0.05)
        CVaR_95_equi = ret_equitativo[ret_equitativo <= VaR_95_equi].mean()

        st.write("VaR al 95%:")
        st.table(pd.DataFrame({
            "VaR": [VaR_95_equi]
        }))
        st.write("CVaR al 95%:")
        st.table(pd.DataFrame({
            "CVaR": [CVaR_95_equi]
        }))





with tab2: 
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import streamlit as st
    
    # Configuración de activos
    etfs = ['TLT', 'EMB', 'SPY', 'EEM', 'GLD']
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    
    # Descargar precios ajustados
    prices = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    returns = prices.pct_change().dropna()
    
    # Calcular rendimientos y covarianzas
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov()
    
    # Función para calcular la volatilidad del portafolio
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Restricciones para la optimización
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(len(etfs))]
     
    # Punto inicial: asignación igualitaria
    initial_weights = np.ones(len(etfs)) / len(etfs)
    
    # Optimización para encontrar pesos óptimos
    result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # Crear una tabla de perspectivas para los activos seleccionados
    perspectives_data = {
        "Activo": ["TLT", "EMB", "SPY", "EEM", "GLD"],
        "Justificación": [
            "Con tasas de interés que podrían mantenerse estables o disminuir ante señales de una desaceleración económica en EE.UU., los bonos del Tesoro a largo plazo podrían ver una apreciación moderada.",
            "Los bonos de mercados emergentes presentan riesgos moderados por volatilidad en las tasas de interés globales y presiones de endeudamiento en economías emergentes, pero ofrecen un rendimiento atractivo.",
            "Las acciones estadounidenses tienen una alta probabilidad de un desempeño mixto, dada la incertidumbre sobre las ganancias corporativas y el crecimiento económico.",
            "Las economías emergentes, lideradas por China y otros países asiáticos, pueden beneficiarse de un panorama más favorable en términos de comercio global, aunque aún enfrentan riesgos geopolíticos.",
            "El oro podría actuar como refugio seguro en un entorno de alta incertidumbre macroeconómica y posibles depreciaciones del dólar."
            ],
            "Racional": [
                "La Reserva Federal ha indicado que podría pausar las subidas de tasas, favoreciendo instrumentos de renta fija a largo plazo. Estimo un rendimiento esperado del 3% anual en USD.",
                "Considerando un entorno de estabilización en la inflación global y spreads más estables, espero un rendimiento del 5% anual en USD.",
                "Aunque el mercado de acciones puede estar limitado por valoraciones altas, sectores defensivos podrían ofrecer estabilidad. Estimo un rendimiento esperado del 6% anual en USD.",
                "Espero un rendimiento de 8% anual en USD, impulsado por una mejora gradual en el crecimiento económico y menores presiones inflacionarias en mercados clave.",
                "Con expectativas de tasas reales estables y una demanda constante, estimo un rendimiento del 4% anual en USD."
                ],
            "Rendimiento esperado": ["3% anual", "5% anual", "6% anual", "8% anual", "4% anual"]
        }
    
    # Crear un DataFrame con las perspectivas
    perspectives_df = pd.DataFrame(perspectives_data)
    
    # Aplicación Streamlit
    st.title("Optimización de Portafolio con Perspectivas de Mercado")
    
    # Mostrar la tabla de perspectivas
    st.subheader("Justificación de perspectivas para los activos seleccionados:")
    st.table(perspectives_df)
    
    # Gráfica de la optimización
    st.subheader("Gráfica de Pesos Óptimos para el Portafolio")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(optimal_weights, labels=etfs, autopct='%1.1f%%', startangle=90, colors=['#581845', '#5733ff', '#ff5733', '#ff33A1', '#C70039'])
    ax.axis('equal')  # Para que la gráfica quede como un círculo
    st.pyplot(fig)
    
    # Gráfica de evolución de precios ajustados
    st.subheader("Evolución de los precios ajustados de los activos")
    fig, ax = plt.subplots(figsize=(10, 6))
    for symbol in etfs:
        ax.plot(prices.index, prices[symbol], label=symbol)
    ax.set_title("Evolución de precios ajustados")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio Ajustado")
    ax.legend()
    st.pyplot(fig)
    
    # Mostrar resultados calculados
    st.subheader("Resultados de la optimización")
    st.write("Pesos óptimos encontrados por optimización con restricciones:")
    optimal_weights_rounded = [round(w * 100, 2) for w in optimal_weights]
    results_df = pd.DataFrame({
        "Activo": etfs,
        "Peso (%)": optimal_weights_rounded
        })
    
    st.table(results_df)
































