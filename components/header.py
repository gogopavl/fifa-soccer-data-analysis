import dash_html_components as html
import dash_core_components as dcc

def Header():
    return html.Div([
        # get_logo(),
        get_header(),
        html.Br([]),
        get_menu()
    ])

def get_logo():
    logo = html.Div([

        html.Div([
            html.Img(src='http://logonoid.com/images/vanguard-logo.png', height='40', width='160')
        ], className="ten columns padded"),

        html.Div([
            dcc.Link('Full View', href='/fifa18/all')
        ], className="two columns page-view no-print")

    ], className="row gs-header")
    return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H5(
                'FIFA WORLD CUP 2018 Qualifiers Data Analysis')
        ], className="twelve columns padded")

    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([

        dcc.Link('Home   ', href='/fifa18/home', className="tab first"),

        dcc.Link('Team Between. & Clos.', href='/fifa18/bet-clo', className="tab"),

        dcc.Link('Team Formation', href='/fifa18/team-formation', className="tab"),

        dcc.Link('Player Comparison', href='/fifa18/pc', className="tab"),

        dcc.Link('Formation Builder', href='/fifa18/builder', className="tab"),

        dcc.Link('Betweenness Flow', href='/fifa18/flow', className="tab")

    ], className="row ")
    return menu
