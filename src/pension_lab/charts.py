"""Charts for the app. Each portfolio keeps one colour and one marker shape everywhere."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PORTFOLIO_COLORS = {
    "a_short_bonds": "#2a78d6",
    "b_long_bonds": "#eb6834",
    "c_growth_mix": "#1baf7a",
}
PORTFOLIO_MARKERS = {
    "a_short_bonds": "circle",
    "b_long_bonds": "square",
    "c_growth_mix": "diamond",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e8e7e3"


def base_layout(height):
    """Return the layout shared by every chart: quiet axes and a clear hover label."""
    return {
        "height": height,
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": "Helvetica, Arial, sans-serif", "size": 13, "color": TEXT},
        "margin": {"l": 10, "r": 10, "t": 40, "b": 40},
        "hoverlabel": {"bgcolor": "white", "font": {"color": TEXT}},
    }


def with_alpha(hex_color, alpha):
    """Return a hex colour as rgba, for the shaded range."""
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def payments_chart(yearly):
    """Return expected yearly payments as thin bars, with one row of data per calendar year."""
    figure = go.Figure(go.Bar(
        x=yearly["calendar_year"], y=yearly["expected_payment"], marker_color=MUTED,
        marker_line_width=0,
        hovertemplate="%{x}: EUR %{y:,.0f}<extra></extra>",
    ))
    figure.update_layout(**base_layout(320), bargap=0.25)
    figure.update_xaxes(title="Calendar year", showgrid=False, color=MUTED)
    figure.update_yaxes(title="Expected payment (EUR)", gridcolor=GRID, color=MUTED,
                        tickformat=",.0f", rangemode="tozero", zeroline=False)
    return figure


def funding_range_chart(bands, floor):
    """Return one panel per portfolio with the P10 to P90 range and the P50 line."""
    portfolio_ids = list(bands["portfolio_id"].unique())
    # Short titles and end-only ticks keep the panels readable at phone width
    labels = ["Portfolio " + bands.loc[bands["portfolio_id"] == pid, "portfolio"].iloc[0][0]
              for pid in portfolio_ids]
    horizon = int(bands["year"].max())
    figure = make_subplots(rows=1, cols=len(portfolio_ids), shared_yaxes=True,
                           subplot_titles=labels, horizontal_spacing=0.07)

    for column, portfolio_id in enumerate(portfolio_ids, start=1):
        rows = bands[bands["portfolio_id"] == portfolio_id]
        color = PORTFOLIO_COLORS[portfolio_id]
        figure.add_trace(go.Scatter(x=rows["year"], y=rows["p90"], mode="lines", line_width=0,
                                    showlegend=False, hoverinfo="skip"), row=1, col=column)
        figure.add_trace(go.Scatter(x=rows["year"], y=rows["p10"], mode="lines", line_width=0,
                                    fill="tonexty", fillcolor=with_alpha(color, 0.22),
                                    showlegend=False, hoverinfo="skip"), row=1, col=column)
        figure.add_trace(go.Scatter(
            x=rows["year"], y=rows["p50"], mode="lines", line={"color": color, "width": 2},
            showlegend=False, customdata=rows[["p10", "p90"]],
            hovertemplate=("Year %{x}<br>P90 %{customdata[1]:.1%}<br>P50 %{y:.1%}"
                           "<br>P10 %{customdata[0]:.1%}<extra></extra>"),
        ), row=1, col=column)
        figure.add_hline(y=floor, line={"color": MUTED, "width": 1, "dash": "dash"}, row=1,
                         col=column)

    figure.update_layout(**base_layout(340))
    figure.update_annotations(font_size=13)
    figure.update_xaxes(title="Year", showgrid=False, color=MUTED, tickvals=[0, horizon])
    figure.update_yaxes(gridcolor=GRID, color=MUTED, tickformat=".0%", zeroline=False)
    figure.update_yaxes(title="Funding level", col=1)
    return figure


def overlapping_stresses(stress):
    """Return the stresses where every portfolio lands on the same funding level."""
    spread = stress.groupby("stress", sort=False)["funding_level"].agg(["min", "max"])
    return spread.index[(spread["max"] - spread["min"]) < 0.0005].tolist()


def stress_chart(stress, floor):
    """Return funding level after each stress as dots, with shape and colour per portfolio."""
    figure = go.Figure()
    for portfolio_id, rows in stress.groupby("portfolio_id", sort=False):
        figure.add_trace(go.Scatter(
            x=rows["funding_level"], y=rows["stress"], mode="markers",
            name=rows["portfolio"].iloc[0],
            marker={"color": PORTFOLIO_COLORS[portfolio_id], "size": 12,
                    "symbol": PORTFOLIO_MARKERS[portfolio_id],
                    "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y}<br>%{fullData.name}: %{x:.1%}<extra></extra>",
        ))
    figure.add_vline(x=floor, line={"color": MUTED, "width": 1, "dash": "dash"})
    layout = base_layout(380)
    layout["margin"]["t"] = 60
    figure.update_layout(**layout, legend={"orientation": "h", "y": 1.12, "x": 0})
    figure.update_xaxes(title="Funding level after the stress", tickformat=".0%", gridcolor=GRID,
                        color=MUTED, zeroline=False)
    figure.update_yaxes(autorange="reversed", showgrid=False, color=TEXT, automargin=True)
    return figure
