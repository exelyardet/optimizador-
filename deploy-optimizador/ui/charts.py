"""Chart generation module for portfolio visualization (Plotly)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

from .theme import (
    CATEGORICAL,
    SEQUENTIAL_BLUE,
    DIVERGING_COLORSCALE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    GRIDLINE,
    color_for,
    style_figure,
)


def plot_efficient_frontier(
    random_portfolios: Dict,
    frontier: Dict,
    portfolios: List[Dict],
    rf: float = 0.02
) -> go.Figure:
    """
    Plot the efficient frontier with random portfolios and optimal points.

    Args:
        random_portfolios: Dict with returns, volatility, sharpe arrays
        frontier: Dict with returns and volatility arrays
        portfolios: List of portfolio dicts with name, returns, volatility
        rf: Risk-free rate

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=random_portfolios['volatility'],
        y=random_portfolios['returns'],
        mode='markers',
        marker=dict(
            size=6,
            color=random_portfolios['sharpe'],
            colorscale=SEQUENTIAL_BLUE,
            showscale=True,
            colorbar=dict(title='Sharpe', thickness=14),
            opacity=0.55,
            line=dict(width=0)
        ),
        name='Portfolios Aleatorios',
        hovertemplate='Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))

    if portfolios:
        max_sharpe_port = max(portfolios, key=lambda x: x.get('sharpe', 0))
        sharpe = max_sharpe_port.get('sharpe', 0)
        vol_max = np.nanmax(frontier['volatility']) * 1.2
        vol_cml = np.linspace(0, vol_max, 100)
        cml_line = rf + sharpe * vol_cml
        fig.add_trace(go.Scatter(
            x=vol_cml, y=cml_line, mode='lines',
            line=dict(color=CATEGORICAL[7], width=1.5, dash='dash'),
            opacity=0.7,
            name='CML',
            hoverinfo='skip'
        ))

    valid_mask = ~np.isnan(frontier['volatility'])
    fig.add_trace(go.Scatter(
        x=frontier['volatility'][valid_mask],
        y=frontier['returns'][valid_mask],
        mode='lines',
        line=dict(color=TEXT_SECONDARY, width=2),
        name='Frontera Eficiente',
        hovertemplate='Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>'
    ))

    # Label offsets (pixels) fan the callouts out around the cluster of
    # optimal points, which often sit very close together on the frontier.
    label_offsets = [(-70, -55), (-70, 55), (90, -55), (90, 55)]
    markers = ['star', 'diamond', 'square', 'triangle-up']
    for i, port in enumerate(portfolios):
        marker_color = CATEGORICAL[(i + 1) % len(CATEGORICAL)]
        fig.add_trace(go.Scatter(
            x=[port['volatility']],
            y=[port['returns']],
            mode='markers',
            marker=dict(
                symbol=markers[i % len(markers)],
                size=14,
                color=marker_color,
                line=dict(color=TEXT_PRIMARY, width=1.5)
            ),
            name=port['name'],
            hovertemplate=f"{port['name']}<br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>"
        ))

        ax, ay = label_offsets[i % len(label_offsets)]
        fig.add_annotation(
            x=port['volatility'], y=port['returns'],
            text=port['name'],
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=marker_color,
            ax=ax, ay=ay,
            font=dict(size=12, color=TEXT_PRIMARY),
            bgcolor='rgba(16,26,46,0.9)',
            bordercolor=marker_color,
            borderwidth=1,
            borderpad=4
        )

    style_figure(
        fig,
        title_text='Espacio de Portfolios (Markowitz)',
        xaxis_title='Volatilidad Anual',
        yaxis_title='Retorno Anual',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        height=580
    )
    fig.update_xaxes(tickformat='.0%')
    fig.update_yaxes(tickformat='.0%')

    return fig


def plot_portfolio_weights(
    weights: np.ndarray,
    assets: List[str],
    title: str,
    color: str = None
) -> go.Figure:
    """
    Plot horizontal bar chart of portfolio weights.

    Args:
        weights: Array of portfolio weights
        assets: List of asset names
        title: Chart title
        color: Unused, kept for backwards-compatible call sites (each asset
            gets its own fixed-order categorical color instead)

    Returns:
        Plotly figure
    """
    df = pd.DataFrame({'Activo': assets, 'Peso (%)': np.array(weights) * 100})
    df = df[df['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=True)

    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text='No hay activos con peso significativo',
            showarrow=False, font=dict(size=14, color=TEXT_SECONDARY)
        )
        style_figure(fig, title_text=title, height=200)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    colors = [color_for(i) for i in range(len(df))]

    fig.add_trace(go.Bar(
        x=df['Peso (%)'],
        y=df['Activo'],
        orientation='h',
        marker=dict(color=colors, line=dict(color=TEXT_PRIMARY, width=0.5)),
        text=[f'{w:.1f}%' for w in df['Peso (%)']],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))

    style_figure(
        fig,
        title_text=title,
        xaxis_title='Peso (%)',
        showlegend=False,
        height=max(280, len(df) * 42)
    )
    fig.update_xaxes(range=[0, max(100, df['Peso (%)'].max() + 12)])

    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    assets: List[str]
) -> Tuple[go.Figure, List[Tuple[str, str, float]]]:
    """
    Plot correlation matrix heatmap.

    Args:
        returns: DataFrame with asset returns
        assets: List of assets to include

    Returns:
        Tuple of (figure, list of high correlations)
    """
    corr_matrix = returns[assets].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=DIVERGING_COLORSCALE,
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=11, color=TEXT_PRIMARY),
        colorbar=dict(title='Correlacion', thickness=14),
        hovertemplate='%{x} - %{y}: %{z:.2f}<extra></extra>'
    ))

    style_figure(
        fig,
        title_text='Matriz de Correlacion entre Activos',
        height=max(400, len(assets) * 55)
    )
    fig.update_yaxes(autorange='reversed')

    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.80:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    return fig, high_corr


def plot_cumulative_returns(
    portfolio_returns: Dict[str, pd.Series],
    benchmark_returns: pd.DataFrame
) -> go.Figure:
    """
    Plot cumulative returns comparison.

    Args:
        portfolio_returns: Dict mapping names to returns series
        benchmark_returns: DataFrame with benchmark returns

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for i, (name, returns) in enumerate(portfolio_returns.items()):
        cum = (1 + returns).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum * 100, mode='lines', name=name,
            line=dict(color=color_for(i), width=2.5),
            hovertemplate=f'{name}<br>%{{x|%Y-%m-%d}}: %{{y:.1f}}%<extra></extra>'
        ))

    n_ports = len(portfolio_returns)
    for j, col in enumerate(benchmark_returns.columns):
        cum = (1 + benchmark_returns[col]).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum * 100, mode='lines', name=col,
            line=dict(color=color_for(n_ports + j), width=1.5, dash='dash'),
            hovertemplate=f'{col}<br>%{{x|%Y-%m-%d}}: %{{y:.1f}}%<extra></extra>'
        ))

    style_figure(
        fig,
        title_text='Rendimientos Acumulados: Portfolios vs Benchmarks',
        xaxis_title='Fecha',
        yaxis_title='Crecimiento Acumulado (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified',
        height=480
    )

    return fig


def plot_cagr_comparison(
    cagr_data: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Plot CAGR comparison bar chart.

    Args:
        cagr_data: Dict with 'assets', 'portfolios', 'benchmarks' subdicts

    Returns:
        Plotly figure
    """
    tipo_colors = {
        'Activo': color_for(0),
        'Portfolio': color_for(1),
        'Benchmark': color_for(2),
    }

    data = []
    for name, cagr in cagr_data.get('assets', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Activo'})
    for name, cagr in cagr_data.get('portfolios', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Portfolio'})
    for name, cagr in cagr_data.get('benchmarks', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Benchmark'})

    df = pd.DataFrame(data)

    fig = go.Figure()
    for tipo, color in tipo_colors.items():
        sub = df[df['Tipo'] == tipo]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub['Nombre'], y=sub['CAGR (%)'], name=tipo,
            marker_color=color,
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ))

    style_figure(
        fig,
        title_text='Comparativa CAGR Anual',
        yaxis_title='CAGR Anual (%)',
        legend_title_text='Tipo',
        height=480
    )
    fig.update_xaxes(tickangle=-45)

    return fig


def plot_tail_risk_histogram_plotly(
    portfolio_returns: Dict[str, pd.Series],
    var_95: Dict[str, float],
    cvar_95: Dict[str, float]
) -> go.Figure:
    """
    Plot histograms of daily returns marking the 95% VaR and CVaR
    (Expected Shortfall) lines to visualize the extreme-loss zone.

    Args:
        portfolio_returns: Dict mapping names to daily returns series
        var_95: Dict mapping names to 95% VaR (positive percentage)
        cvar_95: Dict mapping names to 95% CVaR (positive percentage)

    Returns:
        Plotly figure with one subplot per series
    """
    names = list(portfolio_returns.keys())
    n = len(names)
    cols = 2
    rows = (n + 1) // 2

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)

    for i, name in enumerate(names):
        row, col = divmod(i, cols)
        row, col = row + 1, col + 1
        color = color_for(i)
        data = portfolio_returns[name] * 100

        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=60,
            marker_color=color,
            opacity=0.55,
            histnorm='probability density',
            showlegend=False
        ), row=row, col=col)

        var_val = var_95.get(name, 0)
        cvar_val = cvar_95.get(name, 0)

        fig.add_vline(
            x=-var_val, line_dash='dash', line_color=color, line_width=2,
            annotation_text=f'VaR 95%: {var_val:.2f}%', annotation_position='top left',
            row=row, col=col
        )
        fig.add_vline(
            x=-cvar_val, line_dash='dot', line_color=TEXT_MUTED, line_width=2,
            annotation_text=f'CVaR 95%: {cvar_val:.2f}%', annotation_position='bottom left',
            row=row, col=col
        )

    style_figure(
        fig,
        title_text='Distribucion de Retornos Diarios: Zona de Perdida Extrema (VaR / CVaR 95%)',
        height=380 * rows,
        showlegend=False,
        margin=dict(t=90, l=16, r=16, b=16)
    )
    fig.update_xaxes(title_text='Retorno Diario (%)', gridcolor=GRIDLINE)
    fig.update_yaxes(title_text='Densidad', gridcolor=GRIDLINE)

    return fig


def plot_monte_carlo_fan_chart(
    mc_result: Dict,
    confidence: float = 0.95
) -> go.Figure:
    """
    Plot a Monte Carlo fan chart: median trajectory, confidence bands
    (P5-P95 and P25-P75 by default) and a sample of individual paths.

    Args:
        mc_result: Result dict from core.monte_carlo.simulate_gbm_portfolio
        confidence: Confidence level used for the outer band labels

    Returns:
        Plotly figure
    """
    pct = mc_result['percentiles']
    days = np.arange(len(pct['median']))

    lower_label = f"{(1 - confidence) * 100:.0f}"
    upper_label = f"{confidence * 100:.0f}"
    band_color = CATEGORICAL[0]

    fig = go.Figure()

    for path in mc_result['sample_paths']:
        fig.add_trace(go.Scatter(
            x=days, y=path, mode='lines',
            line=dict(color='rgba(168,179,199,0.15)', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['upper'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=pct['lower'], mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(57,135,229,0.18)',
        name=f'Banda P{lower_label}-P{upper_label}'
    ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['p75'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=pct['p25'], mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(57,135,229,0.38)',
        name='Banda P25-P75'
    ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['median'], mode='lines',
        line=dict(color=band_color, width=3),
        name='Mediana (P50)'
    ))

    fig.add_hline(
        y=mc_result['initial_capital'], line_dash='dot', line_color=TEXT_SECONDARY,
        annotation_text='Capital Inicial', annotation_position='bottom right'
    )

    style_figure(
        fig,
        title_text='Simulacion Monte Carlo: Proyeccion de Valor de Cartera',
        xaxis_title='Dias de Trading',
        yaxis_title='Valor de Cartera (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        height=520
    )

    return fig


def plot_weights_table(
    weights: np.ndarray,
    assets: List[str]
) -> pd.DataFrame:
    """
    Create a DataFrame of portfolio weights for display.

    Args:
        weights: Array of weights
        assets: List of asset names

    Returns:
        DataFrame formatted for display
    """
    df = pd.DataFrame({
        'Activo': assets,
        'Peso (%)': [f'{w*100:.2f}' for w in weights]
    })
    df = df[df['Peso (%)'].astype(float) > 0.01]
    return df.reset_index(drop=True)
