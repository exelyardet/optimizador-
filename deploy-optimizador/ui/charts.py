"""Chart generation module for portfolio visualization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import streamlit as st


plt.style.use('seaborn-v0_8-whitegrid')


def plot_efficient_frontier(
    random_portfolios: Dict,
    frontier: Dict,
    portfolios: List[Dict],
    rf: float = 0.02
) -> plt.Figure:
    """
    Plot the efficient frontier with random portfolios and optimal points.

    Args:
        random_portfolios: Dict with returns, volatility, sharpe arrays
        frontier: Dict with returns and volatility arrays
        portfolios: List of portfolio dicts with name, returns, volatility
        rf: Risk-free rate

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        random_portfolios['volatility'],
        random_portfolios['returns'],
        c=random_portfolios['sharpe'],
        cmap='viridis',
        alpha=0.5,
        s=10,
        label='Portfolios Aleatorios'
    )
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    valid_mask = ~np.isnan(frontier['volatility'])
    ax.plot(
        frontier['volatility'][valid_mask],
        frontier['returns'][valid_mask],
        'b-',
        linewidth=3,
        label='Frontera Eficiente'
    )

    markers = ['*', 'o', 'X', 's', 'D']
    colors = ['gold', 'red', 'green', 'purple', 'orange']
    sizes = [300, 150, 150, 150, 150]

    for i, port in enumerate(portfolios):
        ax.scatter(
            port['volatility'],
            port['returns'],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=sizes[i % len(sizes)],
            label=port['name'],
            edgecolors='black',
            linewidth=1,
            zorder=5
        )

    if portfolios:
        max_sharpe_port = max(portfolios, key=lambda x: x.get('sharpe', 0))
        sharpe = max_sharpe_port.get('sharpe', 0)
        vol_max = np.nanmax(frontier['volatility']) * 1.2
        vol_cml = np.linspace(0, vol_max, 100)
        cml_line = rf + sharpe * vol_cml
        ax.plot(vol_cml, cml_line, 'r--', linewidth=2, label='CML')

    ax.set_title('Espacio de Portfolios (Markowitz)', fontsize=16, weight='bold')
    ax.set_xlabel('Volatilidad Anual', fontsize=12)
    ax.set_ylabel('Retorno Anual', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_portfolio_weights(
    weights: np.ndarray,
    assets: List[str],
    title: str,
    color: str = '#1976D2'
) -> plt.Figure:
    """
    Plot horizontal bar chart of portfolio weights.

    Args:
        weights: Array of portfolio weights
        assets: List of asset names
        title: Chart title
        color: Primary color for the chart

    Returns:
        Matplotlib figure
    """
    df = pd.DataFrame({'Activo': assets, 'Peso (%)': np.array(weights) * 100})
    df = df[df['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=True)

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No hay activos con peso significativo',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))

    bars = ax.barh(df['Activo'], df['Peso (%)'], color=colors, edgecolor='black', height=0.7)

    ax.set_xlim(0, max(100, df['Peso (%)'].max() + 10))
    ax.set_xlabel('Peso (%)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold', pad=15)

    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f'{width:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_ylabel('')

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    assets: List[str]
) -> Tuple[plt.Figure, List[Tuple[str, str, float]]]:
    """
    Plot correlation matrix heatmap.

    Args:
        returns: DataFrame with asset returns
        assets: List of assets to include

    Returns:
        Tuple of (figure, list of high correlations)
    """
    corr_matrix = returns[assets].corr()

    cmap_custom = LinearSegmentedColormap.from_list(
        'CelesteRojoInvert',
        ['lightblue', 'red'],
        N=256
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 10},
        cmap=cmap_custom,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar_kws={'shrink': 0.8, 'pad': 0.02, 'label': 'Correlacion'},
        ax=ax
    )

    ax.set_title('Matriz de Correlacion entre Activos', fontsize=14, weight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.80:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    plt.tight_layout()
    return fig, high_corr


def plot_cumulative_returns(
    portfolio_returns: Dict[str, pd.Series],
    benchmark_returns: pd.DataFrame
) -> plt.Figure:
    """
    Plot cumulative returns comparison.

    Args:
        portfolio_returns: Dict mapping names to returns series
        benchmark_returns: DataFrame with benchmark returns

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1976D2', '#C62828', '#388E3C', '#7B1FA2']
    for i, (name, returns) in enumerate(portfolio_returns.items()):
        cum = (1 + returns).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=name, linewidth=2,
                color=colors[i % len(colors)])

    for col in benchmark_returns.columns:
        cum = (1 + benchmark_returns[col]).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=col, linestyle='--', linewidth=1.5)

    ax.set_title('Rendimientos Acumulados: Portfolios vs Benchmarks', fontsize=14, weight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Crecimiento Acumulado (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cagr_comparison(
    cagr_data: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """
    Plot CAGR comparison bar chart.

    Args:
        cagr_data: Dict with 'assets', 'portfolios', 'benchmarks' subdicts

    Returns:
        Matplotlib figure
    """
    data = []
    for name, cagr in cagr_data.get('assets', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Activo'})
    for name, cagr in cagr_data.get('portfolios', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Portfolio'})
    for name, cagr in cagr_data.get('benchmarks', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Benchmark'})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Nombre', y='CAGR (%)', hue='Tipo', dodge=False, ax=ax)

    ax.set_title('Comparativa CAGR Anual', fontsize=14, weight='bold')
    ax.set_ylabel('CAGR Anual (%)', fontsize=12)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Tipo', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_tail_risk_histogram_plotly(
    portfolio_returns: Dict[str, pd.Series],
    var_95: Dict[str, float],
    cvar_95: Dict[str, float]
) -> go.Figure:
    """
    Plot Plotly histograms of daily returns marking the 95% VaR and CVaR
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
    colors = ['#1976D2', '#C62828', '#388E3C', '#f0ad4e']

    for i, name in enumerate(names):
        row, col = divmod(i, cols)
        row, col = row + 1, col + 1
        color = colors[i % len(colors)]
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
            x=-cvar_val, line_dash='dot', line_color='black', line_width=2,
            annotation_text=f'CVaR 95%: {cvar_val:.2f}%', annotation_position='bottom left',
            row=row, col=col
        )

    fig.update_layout(
        title_text='Distribucion de Retornos Diarios: Zona de Perdida Extrema (VaR / CVaR 95%)',
        template='plotly_white',
        height=380 * rows,
        showlegend=False,
        margin=dict(t=90)
    )
    fig.update_xaxes(title_text='Retorno Diario (%)')
    fig.update_yaxes(title_text='Densidad')

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

    fig = go.Figure()

    for path in mc_result['sample_paths']:
        fig.add_trace(go.Scatter(
            x=days, y=path, mode='lines',
            line=dict(color='rgba(120,120,120,0.15)', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['upper'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=pct['lower'], mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(25,118,210,0.15)',
        name=f'Banda P{lower_label}-P{upper_label}'
    ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['p75'], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=days, y=pct['p25'], mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(25,118,210,0.35)',
        name='Banda P25-P75'
    ))

    fig.add_trace(go.Scatter(
        x=days, y=pct['median'], mode='lines',
        line=dict(color='#0D47A1', width=3),
        name='Mediana (P50)'
    ))

    fig.add_hline(
        y=mc_result['initial_capital'], line_dash='dot', line_color='gray',
        annotation_text='Capital Inicial', annotation_position='bottom right'
    )

    fig.update_layout(
        title_text='Simulacion Monte Carlo: Proyeccion de Valor de Cartera',
        xaxis_title='Dias de Trading',
        yaxis_title='Valor de Cartera (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
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
