import dash
from dash import html

dash.register_page(__name__, path="/drilldown")

layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 2 — Region drilldown"),
        html.P("This page will host linked timeline + event-type breakdown.")
    ])
])