# coding: utf-8

import dash
import dash_core_components as dcc
import dash_html_components as html
import pitch_plotly as pitch
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import base64
import numpy as np
from sklearn.linear_model import LogisticRegression

from components import Header, make_dash_table, print_button

import pandas as pd

path = 'data/dataset.csv'

demo_dict={}
ML_dict={}

original_data = pd.read_csv(path, sep=';' , decimal=',')
teams = original_data.sort_values('team')['team'].unique()
mydata = original_data.groupby(['team','cluster']).mean()
clusterized = original_data.groupby(['cluster']).mean()

teams = original_data.sort_values('team')['team'].unique()
######Fitting Classification algorithm
y = original_data.loc[:,'cluster']
x = original_data.loc[:,'C1':'Y11']
split  = np.rint(0.8*x.shape[0])
X_clean_train = x[:int(split)].values
X_clean_val = x[int(split):].values
y_train = y[:int(split)]
y_val = y[int(split):]
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_clean_train, y_train)
#####
demo_dict={}
old_form={}
ML=[]
for i in range(0,11):
	ML.append([])

position = np.arange(1,12)
choices = ['Betweenness','Closeness', 'X Position','Y Position']

path_fifa = 'img/cup.png'
with open(path_fifa, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
fifa_encoded = "data:image/png;base64," + encoded_string
path_dist = 'img/pca.png'
with open(path_dist, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
dist_encoded = "data:image/png;base64," + encoded_string
path_pos = 'img/cluster_pos.png'
with open(path_pos, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
pos_encoded = "data:image/png;base64," + encoded_string

app = dash.Dash(__name__)
server = app.server

app.config['suppress_callback_exceptions']=True

# read data for tables (one df per table)
df_fund_facts = pd.read_csv('data/df_fund_facts.csv')
df_price_perf = pd.read_csv('data/df_price_perf.csv')
df_current_prices = pd.read_csv('data/df_current_prices.csv')
df_hist_prices = pd.read_csv('data/df_hist_prices.csv')
df_avg_returns = pd.read_csv('data/df_avg_returns.csv')
df_after_tax = pd.read_csv('data/df_after_tax.csv')
df_recent_returns = pd.read_csv('data/df_recent_returns.csv')
df_equity_char = pd.read_csv('data/df_equity_char.csv')
df_equity_diver = pd.read_csv('data/df_equity_diver.csv')
df_expenses = pd.read_csv('data/df_expenses.csv')
df_minimums = pd.read_csv('data/df_minimums.csv')
df_dividend = pd.read_csv('data/df_dividend.csv')
df_realized = pd.read_csv('data/df_realized.csv')
df_unrealized = pd.read_csv('data/df_unrealized.csv')
df_graph = pd.read_csv("data/df_graph.csv")

## Page layouts
home = html.Div([  # page 1

        print_button(),

        html.Div([
            Header(),

            # Row 3
            html.Div([


                html.Div([
                    html.H6(["FIFA WORLD CUP 2018"],
                            className="gs-header gs-table-header padded"),
                    html.Img(src=fifa_encoded,
            		style={'width':"65%", 'height':"55%", "display":"block", "margin-left":"auto", "margin-right":"auto", 'margin-top':'10px'}),
                ], className="six columns"),

                html.Div([
                    html.H6('Data Description',
                            className="gs-header gs-text-header padded"),

                    html.Br([]),

                    html.P("\
                            The dataset consists of 46 columns and 1058 rows. More specifically,\
                            each row contains information and values/measures for a team's performance \
                             in a distinct match. The column headers are:", style={'text-align':'justify', 'margin-top':'10px'}),

                             html.Ul([
                                html.Li(["\
                                    team: national team name \
                                "]),
                                html.Li(["\
                                    cluster: a cluster tag that categorizes the game  \
                                "]),
                                html.Li(["\
                                    C1-C11: mean closeness centrality score for each of the 11 players \
                                "]),
                                html.Li(["\
                                    B1-B11: mean betweenness centrality score for each of the 11 players \
                                "]),
                                html.Li(["\
                                    X1-X11: median X axis position for each of the 11 players \
                                "]),
                                html.Li(["\
                                    Y1-Y11: median Y axis position for each of the 11 players \
                                "]),
                            ], style={'text-align':'justify'}),

                ], className="six columns"),


            ], className="row "),

            # Row 4

            html.Div([

                html.Div([
                    html.H6('About',
                            className="gs-header gs-text-header padded"),
                        html.Br([]),

                        html.P("This project was implemented by: Ioannis Stournaras, Michail Pourpoulakis \
                                and Pavlos Gogousis", style={'text-align':'justify', 'margin-top':'10px'}),

                        html.P("Course Administrators: Benjamin Bach and \
                                David Murray-Rust", style={'text-align':'justify', 'margin-top':'10px'}),

                        html.P("Data Holder: Gian Marco Campagnolo", style={'text-align':'justify', 'margin-top':'10px'}),

                ], className="six columns"),

                html.Div([
                    html.H6("Visualization from PCA Analysis",
                            className="gs-header gs-table-header padded"),

                        html.P("Principal component analysis (PCA) provides intuition to the user on how the cluster are formed.", style={'text-align':'justify', 'margin-top':'25px'}),

                        html.Img(src=dist_encoded,
                		style={'width':"85%", 'height':"85%", "display":"block", "margin-left":"auto",'margin-top':'20px', "margin-right":"auto"}),
                ], className="six columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")


bet_clo = html.Div([  # page 2

        print_button(),

        html.Div([
            Header(),

            # Row 2

            html.Div([

                html.Div([
                    html.H6("Betweenness and Closeness",
                            className="gs-header gs-table-header padded"),
                    html.Div([
                    html.Div([
                		html.Div([
                			html.Label("Pick First Team"),
                			dcc.Dropdown(
                				id='TeamA',
                				options=[{'label':i,'value':i} for i in teams],
                				value = 'Spain')
                			],
                			style={'width': '48%', 'display': 'inline-block'}),
                		html.Div([
                			html.Label("First Team Available Clusters"),
                			dcc.Dropdown(id="ClusterA",
                			value=1)
                			],
                		style={'width': '48%', 'display': 'inline-block'}),
                		]),

                	html.Div([
                		html.Div([
                			html.Label("Pick Second Team"),
                			dcc.Dropdown(
                			id='TeamB',
                			options=[{'label': i, 'value': i} for i in teams],
                			value='Belgium')
                			],
                			style={'width': '48%', 'display': 'inline-block'}),
                		html.Div([
                			html.Label("Second Team Available Clusters"),
                			dcc.Dropdown(id="ClusterB",
                			value=1)
                			],
                		style={'width': '48%', 'display': 'inline-block'}),
                		]),

                    html.P("The diagrams below demonstrate betweenness and closeness \
                    measures for all of the selected teams' players.", style={'text-align':'center', 'margin-top':'10px'}),

                	dcc.Graph(id='Betweenness', config={'displayModeBar': False}, style={"display": "block", "margin-left":"auto", "margin-right":"auto","height":"300", 'width': '80%'}),
                    dcc.Graph(id='Closeness', config={'displayModeBar': False}, style={"display": "block", "margin-left":"auto", "margin-right":"auto","height":"300", 'width': '80%'})
                ]),


                ], className="twelve columns")

            ], className="row "),

            # Row 3

        ], className="subpage")

    ], className="page")


team_formation = html.Div([ # page 3

        print_button(),

        html.Div([

            Header(),

            # Row 1

            html.Div([

                html.Div([
                    html.H6(["Team Formation"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            # Row 2

            html.Div([
        		html.Div([
        			html.Label("Pick First Team"),
        			dcc.Dropdown(
        				id='Team_1',
        				options=[{'label':i,'value':i} for i in teams],
        				value = 'Spain')
        			],
        			style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
            			html.Label("First Team Available Clusters "),
            			dcc.Dropdown(id="Cluster_1",
            			value=5)
            			],
            		style={'width': '48%', 'display': 'inline-block'}),

    		]),
        	html.Div([

                html.Div([
                    html.Label("Pick Second Team"),
        			dcc.Dropdown(
        				id='Team_2',
        				options=[{'label':i,'value':i} for i in teams],
        				value = 'Belgium')
        			],
        			style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
            			html.Label("Second Team Available Clusters"),
            			dcc.Dropdown(id="Cluster_2",
            			value=2)
            			],
            		style={'width': '48%', 'display': 'inline-block'}),

        		]),

    		html.Div([
    			dcc.Graph(id='Football Pitch',config={'displayModeBar': False})],
    		style={'width': '48%', 'display': 'inline-block', 'height':'250'}),
    		html.Div([
    			dcc.Graph(id='Football Pitch2',config={'displayModeBar': False})],
    		style={'width': '48%', 'display': 'inline-block', 'height':'250', 'margin-left':'20px'}),

            html.Div([
                html.Div([
                    html.H6(["Comparison of Selected Players"],
                            className="gs-header gs-table-header padded"),
                        html.P(["Comparing selected player from First Team \
                                to selected player from Second Team. The blue shade corresponds \
                                to the First team's player, while the orange to the Second one's."], style={'text-align':'justify'}),
                        html.Div([
                            dcc.Graph(id='Radar1',config={'displayModeBar': False})],
                			style={'width': '100%','height':'200'}),
                ], className="six columns"),
                html.Div([
                    html.H6(["Comparison of Player to Other Performances"],
                            className="gs-header gs-table-header padded"),
                        html.P(["Comparing performance of selected player from First Team \
                                to the mean performances of the cluster he \
                                belongs to. Blue is mean and orange is player's performance."], style={'text-align':'justify'}),
                        html.Div([
                            dcc.Graph(id='Radar3',config={'displayModeBar': False})],
                			style={'width': '100%', 'height':'200'}),

                ], className="six columns"),
            ], className="row"),



        	# html.Div([
        	# 	# html.Div([
        	# 	# 	dcc.Graph(id='Radar2',config={'displayModeBar': False})],
        	# 	# 	style={'width': '48%', 'display': 'inline-block', 'height':'200'}),
        	# 	html.Div([
        	# 		dcc.Graph(id='Radar3',config={'displayModeBar': False})],
        	# 		style={'width': '48%', 'display': 'inline-block', 'height':'200'}),
        	# 	html.Div([
        	# 		dcc.Graph(id='Radar1',config={'displayModeBar': False})],
        	# 		style={'width': '48%', 'display': 'inline-block', 'height':'200', 'margin-right':'auto'}),
        	# ])

        ], className="subpage")

    ], className="page")

pc = html.Div([  # page 4

        print_button(),

        html.Div([

            Header(),

            # Row 1

            html.Div([

                html.Div([
                    html.H6(["Player Measures Comparison"],
                            className="gs-header gs-table-header padded"),

                            html.Div([
                        		html.Div([
                        			html.Label("Pick First Player Number"),
                        			dcc.Dropdown(
                        				id='Number1',
                        				options=[{'label':i,'value':i} for i in position],
                        				value = 6)
                        			],
                        			style={'width': '48%', 'display': 'inline-block'}),
                        		html.Div([
                        			html.Label("Pick Second Player Number"),
                        			dcc.Dropdown(
                        				id='Number2',
                        				options=[{'label':i,'value':i} for i in position],
                        				value = 2)
                        			],
                        			style={'width': '48%', 'display': 'inline-block'}),
                        		html.Div([
                        			html.Label("Select Measure"),
                        			dcc.Dropdown(
                        				id="Choices",
                        				options=[{'label':i,'value':i} for i in choices],
                        				value='Closeness')
                        			],
                        		style={'width': '48%', 'display': 'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        		]),

                            html.P("Comparing two player positions based on the given metric.", style={'text-align':'center', 'margin-top':'10px'}),

                        	dcc.Graph(id='Stats', config={'displayModeBar': False})


                ], className="twelve columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")

builder = html.Div([  # page 4

        print_button(),

        html.Div([

            Header(),

            # Row 1

            html.Div([

                html.Div([
                    html.H6(["Team Formation Builder and Cluster Prediction"],
                            className="gs-header gs-table-header padded"),
                    html.Div([
                		html.Div([
                			html.Div([
                				html.Label("Choose your Team"),
                				dcc.Dropdown(
                					id='Team_for_Formation',
                					options=[{'label':i,'value':i} for i in teams],
                					value = 'Spain')
                				],
                				style={'width': '33.3%', 'display': 'inline-block'}),
                			html.Div([
                				html.Label("Pick a Cluster"),
                				dcc.Dropdown(id="Cluster_for_Formation",
                				value=1)
                				],
                			style={'width': '33.4%', 'display': 'inline-block'}),
                			html.Div([
                				html.Label("Pick a Game"),
                				dcc.Dropdown(id="Game_for_Formation",
                				value=588)
                				],
                			style={'width': '33.3%', 'display': 'inline-block'}),
                			]),
                		html.Div([
                			dcc.Graph(id='Football_Pitch_Mul',config={'displayModeBar': False})],
                		style={'width': '55%','display': 'inline-block'}),
                	html.Div([
                		html.Label("Select a Player"),
                		dcc.Dropdown(
                			id = 'Player',
                			options =[{'label':"Player %s"%i, 'value':i} for i in range(1,12)],
                			value=1),
                		html.Label("Modify X Position"),
                		dcc.Slider(
                			id='X_slider',
                			min=0,
                			max=100,
                			step=0.1),
                		html.Label("Modify Y position"),
                		dcc.Slider(
                			id='Y_slider',
                			min=0,
                			max=100,
                			#marks={i*10:str(i*10) for i in range(1,11)},
                			step=0.1),
                		html.Label("Modify Betweenness"),
                		dcc.Slider(
                			id='Bet_slider',
                			min=0,
                			max=1,
                			#marks={i/10:str(i/10) for i in range(1,11)},
                			step=0.001),
                		html.Label("Modify Closeness"),
                		dcc.Slider(
                			id='Clo_slider',
                			min=0,
                			max=10,
                			#marks={i:str(i) for i in range(1,11)},
                			step=0.01),
                			html.Button(id ='submit',n_clicks=0, children="Submit Player")],
                		style={'width': '45%','display': 'inline-block'}),
                	])

                ], className="twelve columns"),

            ], className="row "),

            html.Div([
                html.Div([
                    html.H6(["Customized Formation"],
                            className="gs-header gs-table-header padded"),

                        html.Div([
                			dcc.Graph(id="Football_Pitch",config={'displayModeBar': False}),
                			html.Button(id="submit_team",n_clicks=0, children="Submit New Formation", style={'margin-left':'55', 'margin-top':'20px'}),

                		], style= {'width': '50%','display': 'inline-block'})

                ], className="six columns"),
                html.Div([
                    html.H6(["Cluster Prediction Based on Custom Formation"],
                        className="gs-header gs-table-header padded"),

                        html.P(["The first pitch displays the original formation and performance of the selected team.  \
                                We customise formation and performance of players and sumbit each modified instance. \
                                The player's new position and values are set in the second pitch. Lastly, once all changes \
                                have been made, the formation and performances are sumbitted and the system classifies the \
                                playing style of this customisation in a cluster (1-6)."], style={'text-align':'justify', 'margin-top':'20px'}),

                        html.P(id = "initial", style={'font-weight':'bold','margin-top':'10px', 'display':'inline-block','text-align':'justify'}),
                        html.P(id='print', style={'font-weight':'bold','margin-top':'10px', 'display':'inline-block','text-align':'justify'}),

                ], className="six columns"),

            ], className="row"),

        ], className="subpage")

    ], className="page")


flow = html.Div([ # page 3

        print_button(),

        html.Div([

            Header(),

            # Row 1

            html.Div([

                html.Div([
                    html.H6(["Betweenness Flow Within a Team"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            # Row 2

            html.Div([
        		html.Div([
        			html.Label("Pick Team"),
        			dcc.Dropdown(
        				id='Flow_team',
        				options=[{'label':i,'value':i} for i in teams],
        				value = 'Spain')
        			],
        			style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
            			html.Label("Team Available Clusters "),
            			dcc.Dropdown(id="Cluster_flow",
            			value=5)
            			],
            		style={'width': '48%', 'display': 'inline-block'}),
    		]),

            html.P(["The figure below demonstrates the betweenness flow and the closeness of a team. \
                    Players with high betweenness scores are connected with weighted lines, while players with \
                    low betweenness scores are connected with dashed lines. Players with extremely high betweenness \
                    are connected with black bold lines. Scatter markers (circles) are plotted based on the closeness \
                    of each player and their size is proportional to the closeness value (scaled properly)."], style={'text-align':'justify', 'margin-top':'20px'}),

    		html.Div([
    			dcc.Graph(id='Football Pitch Flow',config={'displayModeBar': False})],
    		style={'width': '48%', 'display': 'inline-block', 'height':'250', 'margin-left':'15px'}),

        ], className="subpage")

    ], className="page")

noPage = html.Div([  # 404

    html.P(["404 Page not found"])

    ], className="no-page")



# Describe the layout, or the UI, of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

##########################
##########################
## CUSTOM CALLBACK FUNCTS
##########################
##########################

@app.callback(
	dash.dependencies.Output('ClusterA','options'),
	[dash.dependencies.Input('TeamA', 'value')])
def set_clusterA_options(selected_team):
	avail_clus=mydata.loc[selected_team].index
	return [{'label': i, 'value': i} for i in avail_clus]

@app.callback(
	dash.dependencies.Output('ClusterB','options'),
	[dash.dependencies.Input('TeamB', 'value')])
def set_clusterB_options(selected_team):
	avail_clus=mydata.loc[selected_team].index
	return [{'label': i, 'value': i} for i in avail_clus]

@app.callback(
	dash.dependencies.Output('Closeness', 'figure'),
	[dash.dependencies.Input('TeamA', 'value'),
     dash.dependencies.Input('TeamB', 'value'),
	 dash.dependencies.Input('ClusterA', 'value'),
	 dash.dependencies.Input('ClusterB', 'value')])
def update_graph_Close(A,B,C1,C2):
	data = [
		go.Bar(
			x=mydata.columns[0:11],
			y=mydata.loc[A,C1]["C1":"C11"],
			name=A),

        go.Bar(
			x=mydata.columns[0:11],
			y=mydata.loc[B,C2]["C1":"C11"],
			name=B
		)
	]
	this_layout = go.Layout(
	title = 'Closeness',
		barmode='group'
		)
	return {'data': data, 'layout': this_layout}

@app.callback(
	dash.dependencies.Output('Betweenness', 'figure'),
	[dash.dependencies.Input('TeamA', 'value'),
     dash.dependencies.Input('TeamB', 'value'),
	 dash.dependencies.Input('ClusterA', 'value'),
	 dash.dependencies.Input('ClusterB', 'value')])
def update_graph_Between(A,B,C1,C2):
	data = [
		go.Bar(
			x=mydata.columns[11:22],
			y=mydata.loc[A,C1]["B1":"B11"],
			name=A),

        go.Bar(
			x=mydata.columns[11:22],
			y=mydata.loc[B,C2]["B1":"B11"],
			name=B
		)
	]
	this_layout = go.Layout(
		title = 'Betweenness',
		barmode='group'
		)
	return {'data': data, 'layout': this_layout}


##################################
##################################
### TEAM Formation
##################################
##################################

@app.callback(
	dash.dependencies.Output('Cluster_1','options'),
	[dash.dependencies.Input('Team_1', 'value')])
def set_clusterA_options(selected_team):
	avail_clus=mydata.loc[selected_team].index
	return [{'label': i, 'value': i} for i in avail_clus]

@app.callback(
	dash.dependencies.Output('Cluster_2','options'),
	[dash.dependencies.Input('Team_2', 'value')])
def set_clusterB_options(selected_team):
	avail_clus=mydata.loc[selected_team].index
	return [{'label': i, 'value': i} for i in avail_clus]

@app.callback(
	dash.dependencies.Output('Football Pitch', 'figure'),
	[dash.dependencies.Input('Team_1', 'value'),
	dash.dependencies.Input('Cluster_1', 'value')])
def update_graph_Field(team,cluster):
	title = '%s Formation'%team
	plotter = pitch.Plotter(title)
	demo_arr=[]
	temp_frame = mydata.loc[team,cluster]
	for i in range(1,12):
		player=[temp_frame['X%s' %i],temp_frame['Y%s' %i],\
		'Player %s' %i,13]
		demo_arr.append(player)
	plotter.add_events(demo_arr)
	data, layout = plotter.plot()
	return {'data': data, 'layout': layout}

@app.callback(
dash.dependencies.Output('Radar1', 'figure'),
	[dash.dependencies.Input('Football Pitch', 'hoverData'),
	dash.dependencies.Input('Football Pitch2', 'hoverData'),
	dash.dependencies.Input('Team_1', 'value'),
	dash.dependencies.Input('Cluster_1', 'value'),
	dash.dependencies.Input('Team_2', 'value'),
	dash.dependencies.Input('Cluster_2', 'value')])
def update_graph_Radar(hoverData1,hoverData2,team1,cluster1,team2,cluster2):
    if hoverData1 is None and hoverData2 is None:
        keimeno1="Player 2"
        keimeno2="Player 3"
        n1 = "2"
        n2 = "3"
    else:
        keimeno1 = hoverData1['points'][0]['text']
        n1 = keimeno1.split(" ")[1]
        keimeno2 = hoverData2['points'][0]['text']
        n2 = keimeno2.split(" ")[1]
    temp_frame = mydata.loc[team1,cluster1]
    C, B, X, Y = "C"+n1, "B"+n1, "X"+n1, "Y"+n1
    C_,B_, X_, Y_ = temp_frame[[C,B,X,Y]]
    temp_frame = mydata.loc[team2,cluster2]
    C, B, X, Y = "C"+n2, "B"+n2, "X"+n2, "Y"+n2
    C, B, X, Y = temp_frame[[C,B,X,Y]]
    data =[
        go.Scatterpolar(
            r = [X,  C*10, B*100+5, Y],
            theta = ['X',  'Clo.', 'Bet.', 'Y'],
            fill = 'toself',
            name = 'Team A Player Performance',
            text=keimeno1,
            hoverinfo='text'
        ),
        go.Scatterpolar(
            r = [X_,  C_*10, B_*100+5, Y_],
            theta = ['X', 'Clo.', 'Bet.', 'Y'],
            fill = 'toself',
            name = 'Team B Player Performance',
            text=keimeno2,
            hoverinfo='text'
        )
    ]
    layout = go.Layout(
    # title="Comparing Player "+n1+ " to " +n2,
        margin=dict(l=35,r=35,t=0,b=0),
        polar = dict(
            radialaxis = dict(
            visible = True,
            range = [0, 100]
            )
        ),
        showlegend = False
    )
    return {'data': data, 'layout': layout}

@app.callback(
dash.dependencies.Output('Radar3', 'figure'),
	[dash.dependencies.Input('Football Pitch2', 'hoverData'),
	dash.dependencies.Input('Team_2', 'value'),
	dash.dependencies.Input('Cluster_2', 'value')])
def update_graph_Radar2(hoverData2,team2,cluster2):
    if hoverData2 is None:
        keimeno="Player 3"
        n = "3"
    else:
        keimeno = hoverData2['points'][0]['text']
        n = keimeno.split(" ")[1]
    temp_frame = mydata.loc[team2,cluster2]
    C, B, X, Y = "C"+n, "B"+n, "X"+n, "Y"+n
    C_,B_, X_, Y_ = temp_frame[[C,B,X,Y]]
    C, B, X, Y=clusterized.loc[cluster2][[C,B,X,Y]]
    data =[
        go.Scatterpolar(
            r = [X,  C*10, B*100+5,  Y],
            theta = ['X',  'Clo.', 'Bet.', 'Y'],
            fill = 'toself',
            name = 'Mean performance',
            text=keimeno+" Mean Performance",
            hoverinfo='text'
        ),
        go.Scatterpolar(
            r = [X_,  C_*10, B_*100+5,  Y_],
            theta = ['X',  'Clo.', 'Bet.', 'Y'],
            fill = 'toself',
            name = 'Player Performance',
            text=keimeno+" Performance",
            hoverinfo='text'
            )
        ]
    layout = go.Layout(
        # title="Comparing "+team2+ " Player" +n+ " to average performance of cluster "+ str(cluster2),
        margin=dict(l=35,r=35,t=0,b=0),
        polar = dict(
            radialaxis = dict(
            visible = True,
            range = [0, 100]
            )
        ),
        showlegend = False
    )
    return {'data': data, 'layout': layout}

@app.callback(
	dash.dependencies.Output('Football Pitch2', 'figure'),
	[dash.dependencies.Input('Team_2', 'value'),
	dash.dependencies.Input('Cluster_2', 'value')])
def update_graph_Field(team,cluster):
	title = '%s Formation' %team
	plotter = pitch.Plotter(title)
	demo_arr=[]
	temp_frame = mydata.loc[team,cluster]
	for i in range(1,12):
		player=[temp_frame['X%s' %i],temp_frame['Y%s' %i],\
		'Player %s' %i,13]
		demo_arr.append(player)
	plotter.add_events(demo_arr)
	data, layout = plotter.plot()
	return {'data': data, 'layout': layout}

############################
############################
## Custom Formations
############################

#Dropdowns
@app.callback(
	dash.dependencies.Output('Cluster_for_Formation','options'),
	[dash.dependencies.Input('Team_for_Formation', 'value')])
def set_cluster_options(selected_team):
	avail_clus=original_data.loc[original_data['team']==selected_team]['cluster'].unique()
	return [{'label': i, 'value': i} for i in sorted(avail_clus)]
@app.callback(
	dash.dependencies.Output('Game_for_Formation','options'),
	[dash.dependencies.Input('Team_for_Formation', 'value'),
	dash.dependencies.Input('Cluster_for_Formation', 'value')])
def set_game_options(selected_team,selected_cluster):
	avail_games=original_data.loc[(original_data['team']==selected_team) &(original_data['cluster']==selected_cluster)]
	return [{'label': i+1, 'value': j} for i,j in enumerate(avail_games.index)]

#Figure 2
@app.callback(
	dash.dependencies.Output('Football_Pitch_Mul', 'figure'),
	[dash.dependencies.Input('Game_for_Formation', 'value')])
def update_graph_Field(game):
	global demo_dict,ML
	demo_dict={}
	title = 'Initial Formation'
	plotter = pitch.Plotter(title)
	demo_arr=[]
	temp_frame = original_data.iloc[game]
	for i in range(1,12):
		player=[temp_frame['X%s' %i],temp_frame['Y%s' %i],\
		'Player %s' %i,15]
		ML[i-1]=[temp_frame['C%s' %i],temp_frame['B%s' %i],temp_frame['X%s' %i],temp_frame['Y%s' %i]]
		demo_arr.append(player)
	plotter.add_events(demo_arr)
	data, layout = plotter.plot()
	return {'data': data, 'layout': layout}

###X function
@app.callback(
	dash.dependencies.Output('X_slider','value'),
	[dash.dependencies.Input('Player', 'value'),
	dash.dependencies.Input('Game_for_Formation', 'value')])
def set_X_initial(player,game):
	temp_frame = original_data.iloc[game]
	mean = temp_frame['X%s'%player]
	return mean

@app.callback(
	dash.dependencies.Output('X_slider','min'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	min=original_data['X%s'%player].min()
	return min

@app.callback(
	dash.dependencies.Output('X_slider','max'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	max=original_data['X%s'%player].max()
	return max

###Y functions
@app.callback(
	dash.dependencies.Output('Y_slider','value'),
	[dash.dependencies.Input('Player', 'value'),
	dash.dependencies.Input('Game_for_Formation', 'value')])
def set_clusterA_options(player,game):
	temp_frame=original_data.iloc[game]
	mean = temp_frame['Y%s'%player]
	return mean

@app.callback(
	dash.dependencies.Output('Y_slider','min'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	min=original_data['Y%s'%player].min()
	return min

@app.callback(
	dash.dependencies.Output('Y_slider','max'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	max=original_data['Y%s'%player].max()
	return max
###Between
@app.callback(
	dash.dependencies.Output('Bet_slider','value'),
	[dash.dependencies.Input('Player', 'value'),
	dash.dependencies.Input('Game_for_Formation', 'value')])
def set_clusterA_options(player,game):
	temp_frame = original_data.iloc[game]
	mean = temp_frame['B%s'%player]
	return mean

@app.callback(
	dash.dependencies.Output('Bet_slider','min'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	min=original_data['B%s'%player].min()
	return min

@app.callback(
	dash.dependencies.Output('Bet_slider','max'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	max=original_data['B%s'%player].max()
	return max

###Close
@app.callback(
	dash.dependencies.Output('Clo_slider','value'),
	[dash.dependencies.Input('Player', 'value'),
	dash.dependencies.Input('Game_for_Formation', 'value')])
def set_clusterA_options(player,game):
	temp_frame = original_data.iloc[game]
	mean = temp_frame['C%s'%player]
	return mean

@app.callback(
	dash.dependencies.Output('Clo_slider','min'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	min=original_data['C%s'%player].min()
	return min

@app.callback(
	dash.dependencies.Output('Clo_slider','max'),
	[dash.dependencies.Input('Player', 'value')])
def set_clusterA_options(player):
	max=original_data['C%s'%player].max()
	return max


@app.callback(
    dash.dependencies.Output('Football_Pitch', 'figure'),
    [dash.dependencies.Input('submit', 'n_clicks')],
    [dash.dependencies.State('Game_for_Formation', 'value'),
    dash.dependencies.State('Player', 'value'),
    dash.dependencies.State('X_slider', 'value'),
    dash.dependencies.State('Y_slider', 'value'),
    dash.dependencies.State('Clo_slider', 'value'),
    dash.dependencies.State('Bet_slider', 'value')])
def update_graph_Field(click,game,player1,X,Y,C,B):
    title = 'New Formation'
    global ML
    plotter = pitch.Plotter(title)
    temp_frame = original_data.iloc[game]
    for i in range(1,12):
        player=[temp_frame['X%s' %i],temp_frame['Y%s' %i],\
        'Player %s' %i,15]
        ML[i-1]=[temp_frame['C%s' %i],temp_frame['B%s' %i],temp_frame['X%s' %i],temp_frame['Y%s' %i]]
        old_form[i]=player

    if click==0:
        plotter.add_events_dict(old_form,u'Red')
        data, layout = plotter.plot()
        return {'data': data, 'layout': layout}

    if click>0:
        name='Player %s' %player1
        player = [X,Y,name,15]
        ML[player1-1] = [C,B,X,Y]
        demo_dict[player1]=player
        [old_form.pop(key) for key in list(demo_dict.keys())]
        plotter.add_events_dict(demo_dict,u'Black')
    plotter.add_events_dict(old_form,u'Red')
    data, layout = plotter.plot()
    return {'data': data, 'layout': layout}

@app.callback(
	dash.dependencies.Output('print', 'children'),
	[dash.dependencies.Input("submit_team", 'n_clicks')])
def update_output(clicks):
	if any(len(elem) is 0 for elem in ML) & clicks>0:
		return 'Please submit 11 players first'
	Close, Between, X, Y = zip(*ML)
	Close=np.asarray(Close)
	Between=np.asarray(Between)
	X=np.asarray(X)
	Y=np.asarray(Y)
	data = np.concatenate((Close,Between,X,Y),axis=None).T
	a=logreg.predict([data])[0]
	return "The custom formation of the team has a playing style that is now classified in Cluster: "+ str(a)

@app.callback(
	dash.dependencies.Output('initial', 'children'),
	[dash.dependencies.Input("Cluster_for_Formation", 'value')])
def update_output(clicks):
    return "The team's initial formation is originally classified as having a particular playing style belonging to Cluster: "+str(clicks)


#################################
#################################
## COMPARE TWO teams
#################################

@app.callback(
	dash.dependencies.Output('Stats', 'figure'),
	[dash.dependencies.Input('Number1', 'value'),
	 dash.dependencies.Input('Number2', 'value'),
     	 dash.dependencies.Input('Choices', 'value')])
	 #dash.dependencies.Input('ClusterA', 'value'),
	 #dash.dependencies.Input('ClusterB', 'value')])
def update_graph_Close(N1,N2,C):
	data = [
		go.Bar(
			x=np.arange(1,7),
			y=clusterized[C[0]+str(N1)],
			name=C+' '+str(N1)),
		go.Bar(
			x=np.arange(1,7),
			y=clusterized[C[0]+str(N2)],
			name=C+' '+str(N2))
	]
	layout = go.Layout(
	#title = 'Statistics'
		barmode='group'
		)
	return {'data': data, 'layout': layout}

##################################
##################################
### TEAM Flow Betweenness
##################################
##################################

@app.callback(
	dash.dependencies.Output('Cluster_flow','options'),
	[dash.dependencies.Input('Flow_team', 'value')])
def set_clusterA_options(selected_team):
	avail_clus=mydata.loc[selected_team].index
	return [{'label': i, 'value': i} for i in avail_clus]

@app.callback(
	dash.dependencies.Output('Football Pitch Flow', 'figure'),
	[dash.dependencies.Input('Flow_team', 'value'),
	dash.dependencies.Input('Cluster_flow', 'value')])
def update_graph_Field(team,cluster):
	title = '%s Betweenness Flow'%team
	plotter = pitch.Plotter(title, True)
	demo_arr=[]
	temp_frame = mydata.loc[team,cluster]
	for i in range(1,12):
		player=[temp_frame['X%s' %i],temp_frame['Y%s' %i],\
		'Player %s' %i,13, temp_frame['C%s'%i], temp_frame['B%s'%i]]
		demo_arr.append(player)
	plotter.add_flow_events(demo_arr)
	data, layout = plotter.plot()
	return {'data': data, 'layout': layout}


# Update page
# # # # # # # # #
# detail in depth what the callback below is doing
# # # # # # # # #
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/fifa18' or pathname == '/fifa18/home':
        return home
    elif pathname == '/fifa18/bet-clo':
        return bet_clo
    elif pathname == '/fifa18/team-formation':
        return team_formation
    elif pathname == '/fifa18/pc':
        return pc
    elif pathname == '/fifa18/builder':
        return builder
    elif pathname == '/fifa18/flow':
        return flow
    # elif pathname == '/fifa18/all':
    #     return home, bet_clo, team_formation, pc, builder, flow
    else:
        return noPage

# # # # # # # # #
# detail the way that external_css and external_js work and link to alternative method locally hosted
# # # # # # # # #
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


if __name__ == '__main__':
    app.run_server(debug=True)
