# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as pch
# import matplotlib.image as img

import plotly as py
import itertools
from plotly.graph_objs import *

class Plotter:

    def add_flow_events(self, arr):
        xvals, yvals, names, size, clos, betw = zip(*arr)

        permutations = list(itertools.combinations(range(1,12), 2))

        for first, second in permutations:
            xees = list([xvals[first-1], xvals[second-1]])
            yees = list([yvals[first-1], yvals[second-1]])

            if betw[first-1] > 0.1 and betw[second-1] > 0.1:
                lin = dict(
                    color = 'black',
                    width = 8)
            elif betw[first-1] > 0.075 and betw[second-1] > 0.075:
                lin = dict(
                    color = 'green',
                    width = 3)
            else:
                lin = dict(
                    color = 'blue',
                    dash = 'dot',
                    width = 1)

            linetraces = Scatter(
                x=xees,
                y=yees,
    			showlegend=False,
    			hoverinfo = 'none',
                line = lin
            )
            self.traces.append(linetraces)

        siz = [i * 4 for i in clos]
        trace = Scatter(
            x=xvals,
            y=yvals,
			showlegend=False,
            text=names,
			hoverinfo = 'text',
            mode='markers',
            marker=Marker(
                color=u"red",
                size=siz,
                line=Line(width=2, color='black')
            )
        )
        self.traces.append(trace)

    def add_events(self, arr):
        xvals, yvals, names, size = zip(*arr)

        trace = Scatter(
            x=xvals,
            y=yvals,
			showlegend=False,
            text=names,
			hoverinfo = 'text',
            mode='markers',
            marker=Marker(
                color=u"red",
                size=size,
                line=Line(width=2, color='black')
            )
        )

        self.traces.append(trace)

    def add_events_dict(self, mydict, mycolor):
        arr=[]
        for key in mydict:
            arr.append(mydict[key])
        xvals, yvals, names, size = zip(*arr)

        trace = Scatter(
            x=xvals,
            y=yvals,
			showlegend=False,
            text=names,
			hoverinfo = 'text',
            mode='markers',
            marker=Marker(
                color=mycolor,
                size=size,
                line=Line(width=2, color='black')
            )
        )

        self.traces.append(trace)
    def add_contours(self, X, Y):

        trace_1 = Histogram2dContour(
            x=X,
            y=Y,
			ncontours=3,
			showscale=False,
			opacity=1,
			colorscale="Greys",
			reversescale=True,
			hoverinfo = 'none',
			contours = {
				'showlines':True,
				#'start':1,
				#'end':10,
				'coloring':'lines'
        }
        )

        self.traces.append(trace_1)

    def __init__(self, plot_title, is_flow = False):
        self.traces = []

        trace1 = Scatter(
            showlegend=False,
            x=[50, 50],
            y=[100, 0],
            mode='lines',
            line=Line(width=2, color='black')
        )
        self.traces.append(trace1)
        trace2 = Scatter(
            showlegend=False,
            x=[0, 16, 16, 0],
            y=[80, 80, 20, 20],
            mode='lines',
            line=Line(width=2, color='black'),

        )
        self.traces.append(trace2)

        trace3 = Scatter(
            showlegend=False,
            x=[100, 84, 84, 100],
            y=[80, 80, 20, 20],
            mode='lines',
            line=Line(width=2, color='black')
        )
        self.traces.append(trace3)

        trace4 = Scatter(
            showlegend=False,
            x=[100, 100],
            y=[0, 100],
            mode='lines',
            line=Line(width=2, color='black'),

        )
        self.traces.append(trace4)

        trace5 = Scatter(
            showlegend=False,
            x=[0, 100],
            y=[100, 100],
            mode='lines',
            line=Line(width=2, color='black'),

        )
        self.traces.append(trace5)

        trace6 = Scatter(
            showlegend=False,
            x=[0, 100],
            y=[0, 0],
            mode='lines',
            line=Line(width=2, color='black'),

        )
        self.traces.append(trace6)

        trace7 = Scatter(
            showlegend=False,
            x=[0, 0],
            y=[0, 100],
            mode='lines',
            line=Line(width=2, color='black'),

        )
        self.traces.append(trace7)


        if is_flow:
            wid = 700
            heig = 500
        else:
            wid = 350
            heig = 250

        self.layout = Layout(
            # title=plot_title,
            hovermode='closest',
            autosize=True,
            margin=dict(l=10,r=20,t=20,b=0),
            width=wid,
            height=heig,
			# margin=dict(l=50,r=50,t=50,b=0),
            plot_bgcolor='white',
            xaxis=XAxis(
                range=[0, 100],
                showgrid=False,
                showticklabels=False,
                fixedrange=True
            ),
            yaxis=YAxis(
                range=[0, 100],
                showgrid=False,
                fixedrange=True,
                showticklabels=False
            ),
            annotations=Annotations([
                Annotation(
                    x=0,
                    y=0,
                    xref='x',
                    yref='y',
                    xanchor='left',
                    yanchor='top',
                    showarrow=False
                )
            ])
)

    def plot(self):
        return self.traces, self.layout

def create_normalized_pitch(size=(12.5, 8.0), pcolor='none', ecolor='black'):
    """
    Create figure that has football pitch lines drawn on it.  Figure coordinates are normalized to a 100x100 grid.
    If aspect ratio of figure is not equal to 1.5625 (12.5 in / 8.0 in), the figure width is
    adjusted to satisfy the ratio.
    :param size: Tuple (width, height) of figure dimensions, in inches.
    :param pcolor: Color of football pitch.
    :param ecolor: Color of edges.
    :return: figure and axis objects
    """
    fig_width, fig_height = size
    if float(fig_width)/fig_height != 12.5/8.0:
        fig_width = 12.5/8.0 * fig_height
        print("** Figure size has been adjusted to {:.2f} x {:.2f} inches".format(fig_width, fig_height))
    fig = plt.figure(figsize=(fig_width, fig_height), edgecolor=ecolor, facecolor=pcolor)

    # draw axis -- define coordinates of football pitch
    ax_left = 1.0/12.5
    ax_bottom = 0.6/8.0
    ax_width = 10.5/12.5
    ax_height = 6.8/8.0
    rect = [ax_left, ax_bottom, ax_width, ax_height]
    ax = fig.add_axes(rect, facecolor='green')
    ax.set_xlim(-1.0, 101.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # draw rectangle to represent football pitch
    ax.add_patch(pch.Rectangle((0, 0), 100.0, 100.0, facecolor='none', edgecolor=ecolor, linewidth=3.0))

    # draw half-way line
    linetype = 'w-' if ecolor == 'white' else 'k-'
    ax.plot([50.0, 50.0], [0.0, 100.0], linetype, linewidth=2.0)

    # draw center circle
    #   -  because 'real' field is rectangular and 'normalized' field is square,
    #      circles are actually ellipses
    ax.add_patch(pch.Ellipse((50.0, 50.0), 2*8.71, 2*13.46, facecolor='none', edgecolor=ecolor, linewidth=2.0))

    # draw center dot
    ax.add_patch(pch.Ellipse((50.0, 50.0), 2*0.24, 2*0.37, facecolor=ecolor, edgecolor=ecolor))

    # draw penalty area
    ax.add_patch(pch.Rectangle((0, 20.37), 15.71, 59.26, facecolor='none', edgecolor=ecolor, linewidth=2.0))
    ax.add_patch(pch.Rectangle((84.29, 20.37), 15.71, 59.26, facecolor='none', edgecolor=ecolor, linewidth=2.0))

    # draw penalty spots
    ax.add_patch(pch.Ellipse((10.48, 50.0), 2*0.24, 2*0.37, facecolor=ecolor, edgecolor=ecolor))
    ax.add_patch(pch.Ellipse((89.52, 50.0), 2*0.24, 2*0.37, facecolor=ecolor, edgecolor=ecolor))

    # draw goal areas
    ax.add_patch(pch.Rectangle((0, 36.54), 5.24, 26.94, facecolor='none', edgecolor=ecolor, linewidth=2.0))
    ax.add_patch(pch.Rectangle((94.76, 36.54), 5.24, 26.94, facecolor='none', edgecolor=ecolor, linewidth=2.0))

    # draw goal mouths
    ax.add_patch(pch.Rectangle((-0.95, 44.63), 0.95, 10.76, facecolor='none', edgecolor=ecolor, linewidth=1.5))
    ax.add_patch(pch.Rectangle((100.0, 44.63), 0.95, 10.76, facecolor='none', edgecolor=ecolor, linewidth=1.5))

    # draw left penalty arc
    left_angle_start = 360.0 - np.arccos(5.24/8.71) * 180.0/np.pi
    left_angle_end = np.arccos(5.24/8.71) * 180/np.pi
    ax.add_patch(pch.Arc((10.48, 50.0), 2.0*8.71, 2.0*13.46,
                        theta1=left_angle_start, theta2=left_angle_end,
                        facecolor='none', edgecolor=ecolor, linewidth=2.0))

    # draw right penalty arc
    right_angle_start = 180.0 - np.arccos(5.24/8.71) * 180.0/np.pi
    right_angle_end = 180.0 + np.arccos(5.24/8.71) * 180/np.pi
    ax.add_patch(pch.Arc((89.52, 50.0), 2.0*8.71, 2.0*13.46,
                        theta1=right_angle_start, theta2=right_angle_end,
                        facecolor='none', edgecolor=ecolor, linewidth=2.0))

    # draw corner flag arcs
    ax.add_patch(pch.Arc((0.0, 0.0), 1.90, 2.35, theta1=0.0, theta2=90.0,
                         facecolor='none', edgecolor=ecolor, linewidth=2.0))
    ax.add_patch(pch.Arc((100.0, 0.0), 1.90, 2.35, theta1=90.0, theta2=180.0,
                         facecolor='none', edgecolor=ecolor, linewidth=2.0))
    ax.add_patch(pch.Arc((100.0, 100.0), 1.90, 2.35, theta1=180.0, theta2=270.0,
                         facecolor='none', edgecolor=ecolor, linewidth=2.0))
    ax.add_patch(pch.Arc((0.0, 100.0), 1.90, 2.35, theta1=270.0, theta2=0.0,
                         facecolor='none', edgecolor=ecolor, linewidth=2.0))

    return fig, ax
