from dash import Dash, html, dcc, Input, Output
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, title="Severe Weather & Grid Stress")
server = app.server

app.layout = html.Div(
    className="container",
    children=[
        dcc.Location(id="url"),
        html.Div(className="hero", children=[
            html.H1("Severe Weather & Grid Stress", className="title"),
            html.P(
                "U.S. storm impact on electrical grid performance • 2024",
                className="subtitle"
            ),
            html.Div(className="nav", children=[
                dcc.Link("Overview", href="/", className="navlink", id="nav-overview"),
                dcc.Link("Drilldown", href="/drilldown", className="navlink", id="nav-drilldown"),
                dcc.Link("Distributions", href="/distributions", className="navlink", id="nav-distributions"),
                dcc.Link("ML Insights", href="/ml", className="navlink", id="nav-ml"),
            ]),
        ]),
        dash.page_container,
        html.Div(className="footer", children=[
            "Note: storm/grid relationships are exploratory and association-based (not causal). "
            "All visuals are generated with Python (Dash + Plotly)."
        ])
    ]
)

@app.callback(
    Output("nav-overview", "className"),
    Output("nav-drilldown", "className"),
    Output("nav-distributions", "className"),
    Output("nav-ml", "className"),
    Input("url", "pathname"),
)
def set_active(pathname):
    def cls(active: bool):
        return "navlink active" if active else "navlink"

    return (
        cls(pathname == "/" or pathname is None),
        cls(pathname == "/drilldown"),
        cls(pathname == "/distributions"),
        cls(pathname == "/ml"),
    )

if __name__ == "__main__":
    app.run_server(debug=True)