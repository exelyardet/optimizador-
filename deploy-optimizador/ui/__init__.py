"""UI modules for Streamlit app."""
from .charts import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_cagr_comparison,
    plot_tail_risk_histogram_plotly,
    plot_monte_carlo_fan_chart
)
from .components import (
    display_metrics_table,
    display_tail_risk_table,
    display_monte_carlo_kpis,
    display_stress_table
)
