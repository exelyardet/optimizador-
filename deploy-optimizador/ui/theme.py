"""
Shared design tokens for a consistent, professional look across the app.

Centralizes the color palette, typography, and Plotly layout defaults so
every chart in ui/charts.py draws from the same visual system instead of
ad-hoc colors per function.
"""
import plotly.graph_objects as go

# Fixed-order categorical palette (validated for colorblind-safe adjacent
# contrast). Never cycle/reassign by rank - always index by identity.
CATEGORICAL = [
    '#2a78d6',  # 1 blue      - primary series / "Sharpe Optimo"
    '#eb6834',  # 2 orange    - "Min Volatilidad"
    '#1baf7a',  # 3 aqua      - target / third series
    '#eda100',  # 4 yellow
    '#e87ba4',  # 5 magenta
    '#008300',  # 6 green
    '#4a3aa7',  # 7 violet
    '#e34948',  # 8 red
]

# Sequential single-hue ramp (blue), light -> dark, for magnitude encodings
# (e.g. Sharpe ratio color scale).
SEQUENTIAL_BLUE = [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'
]

# Diverging pair for signed data (e.g. correlation): red (positive) <->
# neutral gray (zero) <-> blue (negative).
DIVERGING_COLORSCALE = [
    [0.0, '#2a78d6'],
    [0.5, '#f0efec'],
    [1.0, '#e34948'],
]

# Chart chrome & ink (light surface)
SURFACE = '#fcfcfb'
PAGE = '#f9f9f7'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'

GOOD = '#0ca30c'
CRITICAL = '#d03b3b'

FONT_FAMILY = 'Inter, "Segoe UI", system-ui, sans-serif'

BASE_LAYOUT = dict(
    template='plotly_white',
    font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=13),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    title_font=dict(size=17, family=FONT_FAMILY, color=TEXT_PRIMARY),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    margin=dict(t=64, l=16, r=16, b=16),
)


def style_figure(fig: go.Figure, **layout_overrides) -> go.Figure:
    """
    Apply the shared visual system (font, surfaces, gridlines) to a figure.

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
