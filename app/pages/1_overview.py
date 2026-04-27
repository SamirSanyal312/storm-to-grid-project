import dash
from dash import html, dcc

dash.register_page(__name__, path="/")

layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 1 — Animated choropleth (coming next)"),
        html.P("This page will host the main animated choropleth + filters + quick KPIs.")
    ])
])