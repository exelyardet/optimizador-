"""
Optimizador de Carteras - BDI Consultora
Aplicacion Streamlit para optimizacion de portfolios usando teoria de Markowitz.
"""
import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from core.data_loader import prepare_data
from core.optimizer import PortfolioOptimizer
from core.metrics import (
    build_metrics_table,
    build_tail_risk_table,
    get_portfolio_returns,
    calculate_cagr
)
from core.stress_test import (
    calculate_portfolio_betas,
    run_stress_test,
    run_historical_stress_test,
    get_stress_interpretation
)
from core.monte_carlo import simulate_gbm_portfolio
from ui.charts import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_cagr_comparison,
    plot_tail_risk_histogram_plotly,
    plot_monte_carlo_fan_chart,
    plot_weights_table
)
from ui.components import (
    display_metrics_table,
    display_tail_risk_table,
    display_monte_carlo_kpis,
    display_stress_table,
    display_high_correlation_warning,
    display_portfolio_summary,
    display_var_interpretation
)


st.set_page_config(
    page_title="Optimizador de Carteras",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Optimizador de Carteras")
st.markdown("*Optimizacion de portfolios usando teoria de Markowitz*")
st.markdown("---")


with st.sidebar:
    st.header("Parametros de Entrada")

    tickers_input = st.text_input(
        "Tickers (separados por coma)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        help="Ingrese hasta 50 tickers separados por coma"
    )

    years = st.slider(
        "Anos de historia",
        min_value=1,
        max_value=20,
        value=5,
        help="Cantidad de anos de datos historicos a analizar"
    )

    rf_rate = st.number_input(
        "Tasa libre de riesgo (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        help="Tasa anual libre de riesgo (ej: 2%)"
    ) / 100

    st.markdown("---")
    st.subheader("Opciones Avanzadas")

    use_target_return = st.checkbox("Usar retorno objetivo")
    target_return = None
    if use_target_return:
        target_return = st.number_input(
            "Retorno objetivo (%)",
            min_value=-50.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        ) / 100

    min_weight_mode = st.radio(
        "Peso minimo por activo",
        options=["Igual para todos los activos", "Personalizado por activo"],
        index=0,
        help="Elija si todos los activos comparten el mismo piso de peso minimo, "
             "o si prefiere definir un minimo distinto para cada activo."
    )

    tickers_preview = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:50]

    min_weight = 0.0
    min_weight_by_ticker = None

    if min_weight_mode == "Igual para todos los activos":
        min_weight = st.number_input(
            "Peso minimo por activo (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            help="Peso minimo que debe tener cada activo (0 = sin minimo)"
        ) / 100
    else:
        st.caption("Defina el peso minimo para cada activo:")
        if len(tickers_preview) == 0:
            st.info("Ingrese primero los tickers para personalizar sus pesos minimos.")
        min_weight_by_ticker = {}
        for t in tickers_preview:
            min_weight_by_ticker[t] = st.number_input(
                f"{t} - peso minimo (%)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                key=f"min_weight_{t}",
                help=f"Peso minimo que debe tener {t} (0 = sin minimo)"
            ) / 100

    st.markdown("---")
    st.subheader("Simulacion Monte Carlo")
    with st.expander("🎲 Parametros Monte Carlo", expanded=False):
        mc_n_sims = st.slider(
            "Numero de simulaciones",
            min_value=1000,
            max_value=10000,
            value=2500,
            step=500,
            help="Cantidad de trayectorias aleatorias a simular"
        )

        mc_n_days = st.slider(
            "Horizonte de proyeccion (dias de trading)",
            min_value=30,
            max_value=756,
            value=252,
            help="252 dias = 1 anio, 756 dias = 3 anios de trading"
        )

        mc_initial_capital = st.number_input(
            "Capital inicial (USD)",
            min_value=100.0,
            value=10000.0,
            step=500.0
        )

        mc_confidence_pct = st.slider(
            "Nivel de confianza Fan Chart (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1,
            help="Define la banda externa del grafico (ej: 95% -> P5 a P95)"
        )
        mc_confidence = mc_confidence_pct / 100

    st.markdown("---")
    optimize_button = st.button("🚀 Optimizar Cartera", type="primary", width="stretch")


tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:50]

if len(tickers) == 0:
    st.warning("Por favor ingrese al menos un ticker.")
    st.stop()


if optimize_button:
    with st.spinner("Descargando datos y optimizando..."):
        try:
            data = prepare_data(tickers, years)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if data['invalid_assets']:
            st.warning(f"Tickers no encontrados: {', '.join(data['invalid_assets'])}")

        if len(data['assets']) < 2:
            st.error("Se necesitan al menos 2 activos validos para optimizar.")
            st.stop()

        if min_weight_mode == "Personalizado por activo" and min_weight_by_ticker:
            min_weight_param = {a: min_weight_by_ticker.get(a, 0.0) for a in data['assets']}
        else:
            min_weight_param = min_weight

        try:
            optimizer = PortfolioOptimizer(
                mean_returns=data['mean_returns'],
                cov_matrix=data['cov_matrix'],
                risk_free_rate=rf_rate,
                min_weight=min_weight_param
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

        port_sharpe = optimizer.optimize_max_sharpe()
        port_min_vol = optimizer.optimize_min_volatility()

        port_target = None
        if use_target_return and target_return is not None:
            port_target = optimizer.optimize_target_return(target_return)
            if port_target is None:
                st.warning(f"No se encontro solucion factible para retorno objetivo {target_return*100:.1f}%")

        frontier = optimizer.calculate_efficient_frontier(n_points=100)
        random_ports = optimizer.generate_random_portfolios(n_portfolios=15000)

        portfolio_list = [
            {'name': 'Sharpe Optimo', 'returns': port_sharpe.returns,
             'volatility': port_sharpe.volatility, 'sharpe': port_sharpe.sharpe},
            {'name': 'Min Volatilidad', 'returns': port_min_vol.returns,
             'volatility': port_min_vol.volatility, 'sharpe': port_min_vol.sharpe}
        ]
        if port_target:
            portfolio_list.append({
                'name': port_target.name,
                'returns': port_target.returns,
                'volatility': port_target.volatility,
                'sharpe': port_target.sharpe
            })

        portfolio_returns = {
            'Sharpe Optimo': get_portfolio_returns(data['returns'], port_sharpe.weights, data['assets']),
            'Min Volatilidad': get_portfolio_returns(data['returns'], port_min_vol.weights, data['assets'])
        }
        if port_target:
            portfolio_returns[port_target.name] = get_portfolio_returns(
                data['returns'], port_target.weights, data['assets']
            )

        benchmark_returns = data['returns'][data['benchmarks']]

        st.session_state['optimization_done'] = True
        st.session_state['data'] = data
        st.session_state['optimizer'] = optimizer
        st.session_state['port_sharpe'] = port_sharpe
        st.session_state['port_min_vol'] = port_min_vol
        st.session_state['port_target'] = port_target
        st.session_state['frontier'] = frontier
        st.session_state['random_ports'] = random_ports
        st.session_state['portfolio_list'] = portfolio_list
        st.session_state['portfolio_returns'] = portfolio_returns
        st.session_state['benchmark_returns'] = benchmark_returns
        st.session_state['rf_rate'] = rf_rate


if st.session_state.get('optimization_done', False):
    data = st.session_state['data']
    port_sharpe = st.session_state['port_sharpe']
    port_min_vol = st.session_state['port_min_vol']
    port_target = st.session_state['port_target']
    frontier = st.session_state['frontier']
    random_ports = st.session_state['random_ports']
    portfolio_list = st.session_state['portfolio_list']
    portfolio_returns = st.session_state['portfolio_returns']
    benchmark_returns = st.session_state['benchmark_returns']
    rf_rate = st.session_state['rf_rate']

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Frontera Eficiente",
        "⚖️ Composicion",
        "📈 Metricas",
        "🔗 Correlacion",
        "📉 Rendimiento",
        "⚠️ Riesgo (VaR)",
        "🔥 Stress Test",
        "🎲 Monte Carlo"
    ])

    with tab1:
        st.subheader("Espacio de Portfolios (Markowitz)")
        fig = plot_efficient_frontier(random_ports, frontier, portfolio_list, rf_rate)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            display_portfolio_summary(
                "Sharpe Optimo",
                port_sharpe.returns,
                port_sharpe.volatility,
                port_sharpe.sharpe
            )
        with col2:
            display_portfolio_summary(
                "Min Volatilidad",
                port_min_vol.returns,
                port_min_vol.volatility,
                port_min_vol.sharpe
            )
        if port_target:
            with col3:
                display_portfolio_summary(
                    port_target.name,
                    port_target.returns,
                    port_target.volatility,
                    port_target.sharpe
                )

    with tab2:
        st.subheader("Composicion de Portfolios")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Portfolio Sharpe Optimo")
            fig = plot_portfolio_weights(port_sharpe.weights, data['assets'], "Sharpe Optimo")
            st.pyplot(fig)

            weights_df = plot_weights_table(port_sharpe.weights, data['assets'])
            st.dataframe(weights_df, width="stretch", hide_index=True)

        with col2:
            st.markdown("### Portfolio Minima Volatilidad")
            fig = plot_portfolio_weights(port_min_vol.weights, data['assets'], "Min Volatilidad", "#C62828")
            st.pyplot(fig)

            weights_df = plot_weights_table(port_min_vol.weights, data['assets'])
            st.dataframe(weights_df, width="stretch", hide_index=True)

        if port_target:
            st.markdown(f"### {port_target.name}")
            fig = plot_portfolio_weights(port_target.weights, data['assets'], port_target.name, "#388E3C")
            st.pyplot(fig)

            weights_df = plot_weights_table(port_target.weights, data['assets'])
            st.dataframe(weights_df, width="stretch", hide_index=True)

    with tab3:
        st.subheader("Metricas Comparativas")

        metrics_df = build_metrics_table(portfolio_returns, benchmark_returns, rf_rate)
        display_metrics_table(metrics_df)

        st.markdown("""
        **Interpretacion:**
        - **Retorno Anual**: Rendimiento anualizado del portfolio
        - **Volatilidad**: Desviacion estandar anualizada (riesgo)
        - **Sharpe**: Retorno ajustado por riesgo (mayor es mejor)
        - **CAGR**: Tasa de crecimiento anual compuesta
        - **Beta**: Sensibilidad respecto al benchmark
        """)

    with tab4:
        st.subheader("Matriz de Correlacion")

        fig, high_corr = plot_correlation_matrix(data['returns'], data['assets'])
        st.pyplot(fig)

        display_high_correlation_warning(high_corr)

    with tab5:
        st.subheader("Rendimientos Acumulados")

        fig = plot_cumulative_returns(portfolio_returns, benchmark_returns)
        st.pyplot(fig)

        st.subheader("Comparativa CAGR")

        cum_assets = (1 + data['returns'][data['assets']]).cumprod() - 1
        cum_ports = {name: (1 + ret).cumprod() - 1 for name, ret in portfolio_returns.items()}
        cum_bench = (1 + benchmark_returns).cumprod() - 1

        cagr_data = {
            'assets': {a: calculate_cagr(cum_assets[a]) for a in data['assets']},
            'portfolios': {name: calculate_cagr(cum) for name, cum in cum_ports.items()},
            'benchmarks': {b: calculate_cagr(cum_bench[b]) for b in benchmark_returns.columns}
        }

        fig = plot_cagr_comparison(cagr_data)
        st.pyplot(fig)

    with tab6:
        st.subheader("Metricas de Cola: VaR y CVaR (Expected Shortfall)")

        spy_series = benchmark_returns['SPY'] if 'SPY' in benchmark_returns else benchmark_returns.iloc[:, 0]

        st.markdown("#### Metodo Parametrico (Distribucion Normal)")
        tail_param_df = build_tail_risk_table(portfolio_returns, spy_series, method='parametric')
        display_tail_risk_table(tail_param_df)

        st.markdown("#### Metodo Historico (Empirico)")
        tail_hist_df = build_tail_risk_table(portfolio_returns, spy_series, method='historical')
        display_tail_risk_table(tail_hist_df)

        st.caption(
            "CVaR (Expected Shortfall) = perdida promedio esperada en el peor "
            "5% (o 1%) de los escenarios, es decir E[R | R <= -VaR]."
        )

        st.markdown("---")
        st.subheader("Distribucion de Retornos Diarios: Zona de Perdida Extrema")

        returns_with_spy = portfolio_returns.copy()
        returns_with_spy['SPY'] = spy_series

        var_95_map = dict(zip(tail_hist_df['Portfolio'], tail_hist_df['VaR 95% (%)']))
        cvar_95_map = dict(zip(tail_hist_df['Portfolio'], tail_hist_df['CVaR 95% (%)']))

        fig = plot_tail_risk_histogram_plotly(returns_with_spy, var_95_map, cvar_95_map)
        st.plotly_chart(fig, width="stretch")

        display_var_interpretation()

    with tab7:
        st.subheader("Stress Testing")

        betas = calculate_portfolio_betas(
            portfolio_returns,
            benchmark_returns['SPY'] if 'SPY' in benchmark_returns else benchmark_returns.iloc[:, 0]
        )

        st.markdown("### Escenarios Hipoteticos")
        stress_df = run_stress_test(betas)
        display_stress_table(stress_df)

        st.markdown("---")
        st.markdown("### Escenarios Historicos")
        hist_stress_df = run_historical_stress_test(betas)
        display_stress_table(hist_stress_df)

        st.info(get_stress_interpretation())

    with tab8:
        st.subheader("Simulacion de Monte Carlo (GBM Multi-Activo Correlacionado)")

        portfolio_options = {
            'Sharpe Optimo': port_sharpe,
            'Min Volatilidad': port_min_vol
        }
        if port_target:
            portfolio_options[port_target.name] = port_target

        selected_port_name = st.selectbox(
            "Cartera a simular",
            options=list(portfolio_options.keys()),
            help="Se simula usando los pesos optimos de la cartera elegida"
        )
        selected_port = portfolio_options[selected_port_name]

        with st.spinner("Simulando trayectorias Monte Carlo..."):
            mc_result = simulate_gbm_portfolio(
                mean_returns=data['mean_returns'],
                cov_matrix=data['cov_matrix'],
                weights=selected_port.weights,
                n_sims=mc_n_sims,
                n_days=mc_n_days,
                initial_capital=mc_initial_capital,
                confidence=mc_confidence
            )

        fig = plot_monte_carlo_fan_chart(mc_result, mc_confidence)
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Metricas Terminales")
        display_monte_carlo_kpis(mc_result)

        st.info(f"""
        **Interpretacion:** Se simularon **{mc_n_sims:,}** trayectorias de **{mc_n_days}**
        dias de trading para la cartera **{selected_port_name}**, usando Movimiento
        Browniano Geometrico correlacionado (descomposicion de Cholesky sobre la matriz
        de covarianza anualizada de {', '.join(data['assets'])}). Con un
        **{mc_confidence_pct}% de confianza**, el valor final de la cartera se ubicara
        dentro de la banda mostrada en el grafico (P{100 - mc_confidence_pct} a P{mc_confidence_pct}).
        """)

else:
    st.info("👈 Configure los parametros en el panel lateral y presione **Optimizar Cartera** para comenzar.")

    st.markdown("""
    ### Como usar esta herramienta:

    1. **Ingrese los tickers** de los activos que desea analizar (separados por coma)
    2. **Seleccione el periodo** de historia a analizar
    3. **Configure la tasa libre de riesgo** (default 2%)
    4. *Opcional*: Configure un retorno objetivo o peso minimo por activo
    5. Presione **Optimizar Cartera**

    ### Que obtendra:

    - **Frontera Eficiente**: Visualizacion del espacio riesgo-retorno
    - **Composicion Optima**: Pesos recomendados para cada portfolio
    - **Metricas Comparativas**: Sharpe, CAGR, Beta vs benchmarks
    - **Analisis de Correlacion**: Diversificacion de la cartera
    - **VaR y CVaR**: Estimacion de perdidas potenciales y de cola (Expected Shortfall)
    - **Stress Testing**: Comportamiento ante crisis
    - **Monte Carlo**: Proyeccion probabilistica del valor futuro de la cartera
    """)


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Desarrollado con Streamlit | Datos de Yahoo Finance"
    "</div>",
    unsafe_allow_html=True
)
