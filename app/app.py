from dash import Dash, html, dcc
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, title="Storm → Grid (Python)")
server = app.server

app.layout = html.Div(
    className="container",
    children=[
        html.Div(className="hero", children=[
            html.H1("Storm → Grid", className="title"),
            html.P(
                "A Python-only interactive story connecting storm activity with grid stress. "
                "Overview-first → filter → details-on-demand.",
                className="subtitle"
            ),
            html.Div(className="nav", children=[
                dcc.Link("Overview", href="/", className="navlink"),
                dcc.Link("Drilldown", href="/drilldown", className="navlink"),
                dcc.Link("Distributions", href="/distributions", className="navlink"),
            ]),
        ]),
        dash.page_container
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)