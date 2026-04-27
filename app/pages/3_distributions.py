import dash
from dash import html

dash.register_page(__name__, path="/distributions")

layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 3 — Distributions & extremes"),
        html.P("This page will host log-binned histogram + ECDF/CCDF + outlier table.")
    ])
])