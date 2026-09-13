"""UI components for Streamlit app."""
import pandas as pd
import streamlit as st
from typing import List, Tuple


def display_metrics_table(df: pd.DataFrame) -> None:
    """
    Display styled metrics comparison table.

    Args:
        df: DataFrame with portfolio metrics
    """
    styled = df.style.format({
        'Retorno Anual (%)': '{:.2f}',
        'Volatilidad (%)': '{:.2f}',
        'Sharpe': '{:.2f}',
        'CAGR (%)': '{:.2f}',
        'Retorno 12m (%)': '{:.2f}',
        'Beta SPY': '{:.2f}',
        'Beta QQQ': '{:.2f}'
    }).background_gradient(
        subset=['Retorno Anual (%)'], cmap='Greens'
    ).background_gradient(
        subset=['Volatilidad (%)'], cmap='Reds_r'
    ).background_gradient(
        subset=['Sharpe'], cmap='YlGn'
    ).background_gradient(
        subset=['CAGR (%)'], cmap='Greens'
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch", hide_index=True)


def display_tail_risk_table(df: pd.DataFrame) -> None:
    """
    Display the comparative VaR/CVaR tail-risk table.

    Args:
        df: DataFrame with columns Portfolio, VaR 95% (%), CVaR 95% (%),
            VaR 99% (%), CVaR 99% (%)
    """
    cols = ['VaR 95% (%)', 'CVaR 95% (%)', 'VaR 99% (%)', 'CVaR 99% (%)']
    styled = df.style.format({c: '{:.2f}' for c in cols}).background_gradient(
        subset=cols, cmap='OrRd'
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch", hide_index=True)


def display_monte_carlo_kpis(mc_result: dict) -> None:
    """
    Display terminal Monte Carlo KPIs in a row of metrics.

    Args:
        mc_result: Result dict from core.monte_carlo.simulate_gbm_portfolio
    """
    initial_capital = mc_result['initial_capital']
    median_terminal = mc_result['median_terminal']
    var_mc = mc_result['var_mc']
    cvar_mc = mc_result['cvar_mc']
    prob_loss = mc_result['prob_loss']
    median_dd = mc_result['median_max_drawdown']

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Valor Mediano Esperado",
            f"${median_terminal:,.0f}",
            f"{(median_terminal / initial_capital - 1) * 100:+.1f}%"
        )
    with col2:
        st.metric(
            "Peor Escenario (VaR MC)",
            f"${var_mc:,.0f}",
            f"{(var_mc / initial_capital - 1) * 100:+.1f}%"
        )
    with col3:
        st.metric(
            "CVaR Monte Carlo",
            f"${cvar_mc:,.0f}",
            f"{(cvar_mc / initial_capital - 1) * 100:+.1f}%"
        )
    with col4:
        st.metric("Probabilidad de Perdida", f"{prob_loss * 100:.1f}%")
    with col5:
        st.metric("Max Drawdown Mediano", f"{median_dd * 100:.1f}%")


def display_stress_table(df: pd.DataFrame, title: str = "") -> None:
    """
    Display stress test results table.

    Args:
        df: DataFrame with stress test results
        title: Optional title for the table
    """
    if title:
        st.subheader(title)

    styled = df.style.format('{:.2f}%').background_gradient(
        axis=None, cmap='OrRd', vmin=df.min().min(), vmax=0
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch")


def display_weights_table(weights_df: pd.DataFrame) -> None:
    """
    Display portfolio weights table.

    Args:
        weights_df: DataFrame with asset weights
    """
    st.dataframe(
        weights_df,
        width="stretch",
        hide_index=True
    )


def display_high_correlation_warning(high_corr: List[Tuple[str, str, float]]) -> None:
    """
    Display warning for high correlations.

    Args:
        high_corr: List of (asset1, asset2, correlation) tuples
    """
    if high_corr:
        st.error("**ATENCION: DIVERSIFICACION INSUFICIENTE**")
        st.warning("Se detectaron activos con correlacion > 0.80:")
        for a1, a2, corr in high_corr:
            st.write(f"- **{a1}** - **{a2}**: correlacion {corr:.2f}")
    else:
        st.success("**DIVERSIFICACION CORRECTA** - Ninguna correlacion > 0.80")


def display_portfolio_summary(
    name: str,
    returns: float,
    volatility: float,
    sharpe: float
) -> None:
    """
    Display portfolio summary metrics in columns.

    Args:
        name: Portfolio name
        returns: Annual returns
        volatility: Annual volatility
        sharpe: Sharpe ratio
    """
    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retorno Anual", f"{returns*100:.2f}%")
    with col2:
        st.metric("Volatilidad", f"{volatility*100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")


def display_var_interpretation() -> None:
    """Display interpretation help for VaR/CVaR charts."""
    st.info("""
    **Como interpretar cada grafico:**
    - Cada histograma muestra la frecuencia de retornos diarios del portfolio.
    - La linea discontinua marca el **Value at Risk (VaR)** al 95%: solo hay un 5% de
      chances de que la perdida diaria sea peor que ese valor.
    - La linea punteada marca el **CVaR / Expected Shortfall**: la perdida promedio
      esperada quedando dentro de ese 5% peor de los escenarios (siempre >= VaR).
    - Ejemplo: si el VaR es 2.5% y el CVaR es 3.4%, con 95% de confianza no se espera
      perder mas de 2.5% en un dia, pero si ese limite se supera, la perdida promedio
      en esos casos extremos ronda el 3.4%.
    """)
