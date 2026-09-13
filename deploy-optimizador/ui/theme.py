"""
Shared design tokens for a distinctive, professional dark theme across the app.

Centralizes the color palette, typography, and Plotly layout defaults so
every chart in ui/charts.py and every table in ui/components.py draws from
the same navy dark visual system instead of ad-hoc colors per function.
"""
import plotly.graph_objects as go

# Fixed-order categorical palette (dark-surface steps of a validated
# colorblind-safe set). Never cycle/reassign by rank - always index by identity.
CATEGORICAL = [
    '#3987e5',  # 1 blue      - primary series / "Sharpe Optimo"
    '#d9772e',  # 2 orange    - "Min Volatilidad"
    '#22b884',  # 3 aqua      - target / third series
    '#e0ab2e',  # 4 yellow
    '#e08bb0',  # 5 magenta
    '#3ecf6e',  # 6 green
    '#9085e9',  # 7 violet
    '#e66767',  # 8 red
]

# Sequential single-hue ramp (blue), dark -> light, for magnitude encodings
# on a dark surface: low values recede toward the dark background, high
# values pop bright (the inverse ordering of a light-surface ramp).
SEQUENTIAL_BLUE = [
    '#0d366b', '#184f95', '#1c5cab', '#256abf', '#3987e5', '#6da7ec', '#b7d3f6'
]

# Diverging pair for signed data (e.g. correlation): red (positive) <->
# neutral dark gray (zero) <-> blue (negative), tuned for the dark surface.
DIVERGING_COLORSCALE = [
    [0.0, '#3987e5'],
    [0.5, '#31405c'],
    [1.0, '#e66767'],
]

# Chart chrome & ink (dark navy surface)
PAGE = '#0a1120'
SURFACE = '#101a2e'
SIDEBAR = '#0d1626'
TEXT_PRIMARY = '#f5f7fa'
TEXT_SECONDARY = '#a8b3c7'
TEXT_MUTED = '#6b7690'
GRIDLINE = '#22304a'
BASELINE = '#33445f'

GOOD = '#3ecf6e'
CRITICAL = '#e66767'

FONT_FAMILY = 'Inter, "Segoe UI", system-ui, sans-serif'
DISPLAY_FONT_FAMILY = '"Space Grotesk", Inter, "Segoe UI", system-ui, sans-serif'

BASE_LAYOUT = dict(
    template='plotly_dark',
    font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=13),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    title_font=dict(size=17, family=DISPLAY_FONT_FAMILY, color=TEXT_PRIMARY),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11, color=TEXT_SECONDARY)),
    margin=dict(t=64, l=16, r=16, b=16),
)


def style_figure(fig: go.Figure, **layout_overrides) -> go.Figure:
    """
    Apply the shared dark visual system (font, surfaces, gridlines) to a figure.

    Args:
        fig: Plotly figure to style in place
        layout_overrides: Any go.Figure.update_layout kwargs to override
            the shared defaults (e.g. title_text, xaxis_title)

    Returns:
        The same figure, styled
    """
    layout = dict(BASE_LAYOUT)
    layout.update(layout_overrides)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    return fig


def color_for(index: int) -> str:
    """Return the fixed-order categorical color for series index `index`."""
    return CATEGORICAL[index % len(CATEGORICAL)]
