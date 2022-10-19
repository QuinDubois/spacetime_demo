import numpy as np
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Optional, Union, Tuple

import statsmodels.api as sm
import datetime
import math

from spacetime.graphics.controlCubeSorter import sort_cube_data
from spacetime.operations.cubeToDataframe import cube_to_dataframe

# Style color presets
COLOR_STYLES = {
    "chart_background": "#222831",
    "chart_grid": "#41434a",
    "tick_font": "#bbbbbb",
    "font": "#bbbbbb",
    "line_colors": [
        "#777777"
    ],
    "marker_colors": [
        "#FF00FF",
        "#00FFFF",
        "#0000FF",
        "#FFFF00",
        "#00FF00",
        "#FF0000",
    ]
}

# flags for data styling.
FLAGS = {
    "base": ["Base", COLOR_STYLES["line_colors"][0]],
    "below_avg": ["Below Average", COLOR_STYLES["marker_colors"][0]],
    "above_avg": ["Above Average", COLOR_STYLES["marker_colors"][1]],
    "deviation_above": ["Deviation Below", COLOR_STYLES["marker_colors"][2]],
    "deviation_below": ["Deviation Above", COLOR_STYLES["marker_colors"][3]],
    "trending_up": ["Trending Up", COLOR_STYLES["marker_colors"][4]],
    "trending_down": ["Trending Down", COLOR_STYLES["marker_colors"][5]],
}


# Main cube plotting method
########################################################################################################################
def plot_cube(
        cube,
        plot_type: str = "timeseries",
        variable: Optional[Union[str, int]] = None,
        summary: str = "mean",
        show_avg: str = "all",
        show_deviations: str = "all",
        deviation_coefficient: int = 1,
        show_trends: str = "updown",
        histo_type: str = "value",
        histo_highlight: str = 'variable',
        discrete_latlong_size: Union[int, float] = 10,
        bin_size: Union[int, float] = 100,
        show_plot: bool = True,
) -> go.Figure:

    """
    Parameter Definitions:

        cube: <accepted types: cube object>
                A cube object.

        plot_type: <accepted types: string>
                The type of plot to output.
                Options:
                    'space' - creates a choropleth heatmap
                    'timeseries' - creates a line plot
                    'control' - creates a configurable control chart plot
                    'histogram' - creates a configurable histogram plot
                    'box' - creates a box plot

        variable: <accepted types: string, integer>
                The variable name to filter the dataset by.

        summary: <accepted types: string>
                The aggregation function for the dataset.
                Options:
                    'min' - aggregates by the minimum value
                    'max' - aggregates by the maximum value
                    'median' - aggregates by the median value
                    'mean' - aggregates by the mean value

        show_avg: <accepted types: string>
                For use with Control Charts, allows toggling of highlighting for average values.
                Options:
                    'above' - highlights markers above average
                    'below' - highlights markers below average
                    'all' - combines 'above' and 'below' options
                    'none' - no average based highlighting

        show_deviations: <accepted types: string>
                For use with Control Charts, allows toggling of highlighting for standard deviation
                values. Related: deviation_coefficient
                Options:
                    'above' - highlights markers in positive standard deviation
                    'below' - highlights markers in negative standard deviation
                    'all' - combines 'above' and 'below' options
                    'none' - no deviation based highlighting

        deviation_coefficient: <accepted types: integer>
                For use with Control Charts, set how many standard deviations outside the data set
                normal you want to count for show_deviations highlighting. Related: show_deviations

        show_trends: <accepted types: string>
                For use with Control Charts, allows toggling of trendlines.
                Options:
                    'all' - show all trendlines regardless of p-value
                    'up' - show p-value significant trendlines for positive trends
                    'down' - show p-value significant trendlines for negative trends
                    'updown' - combines 'up' and 'down' options
                    'none' - show no trendlines

        histo_type: <accepted types: string>
                For use with Histograms, determines histogram output type. Related: bin_size
                Options:
                    'single' - a single histogram chart
                    'multi' - a histogram chart for each unique variable in the data set
                    'animated' - a histogram chart that animates over the time series of the data set

        histo_highlight: <accepted types: string>
                For use with Histograms, determines additional highlighting for histogram charts.
                Related: discrete_latlong_size
                Options:
                    'variable' - highlights bins based on which variable the value belongs to.
                                    alias: 'var', 'variables'
                    'latitude' - highlights bins based on the latitude degree range the value was measured in.
                                    alias: 'lat'
                    'longitude' - highlights bins based on the longitude degree range the value was measured in.
                                    alias: 'lon', 'long'

        discrete_latlong_size: <accepted types: integer, float>
                For use with Histograms, determines the size in degrees of the distinctions for latitude and
                longitude based highlighting. Related: histo_highlight

        bin_size: <accepted types: integer, float>
                For use with Histograms, determined the bin size of animated histograms. Related: histo_type

        show_plot: <accepted types: boolean>
                Allows the user to turn off automatic chart output.
    """

    df_plot = organize_dataframe(cube, plot_type, variable, summary)

    input_validity = validate_inputs(df_plot,
                                     plot_type,
                                     variable,
                                     summary,
                                     show_avg,
                                     show_deviations,
                                     show_trends,
                                     histo_type,
                                     histo_highlight,
                                     deviation_coefficient,
                                     discrete_latlong_size,
                                     bin_size,
                                     show_plot,
                                     )

    if input_validity is True:
        fig = go.Figure

        if plot_type == 'space':
            fig = plot_spatial(cube, df_plot)

        elif plot_type == 'timeseries':
            fig = plot_timeseries(df_plot)

        elif plot_type == 'control':
            fig = plot_control(df_plot, show_avg, show_deviations, deviation_coefficient, show_trends)

        elif plot_type == 'histogram':
            fig = plot_histogram(df_plot, histo_type, histo_highlight, discrete_latlong_size, bin_size)

        elif plot_type == 'box':
            fig = plot_box(df_plot, variable)

        if show_plot:
            fig.show()

    return fig


# Secondary cube plotting methods
########################################################################################################################
# Plot a spatial choropleth chart
def plot_spatial(cube, df) -> go.Figure:
    time = df["timeChar"]
    maxVal = np.nanmax(df["value"])

    coords = cube.upper_left_corner()

    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='value',
        animation_frame=time,
        range_color=(0, maxVal),
        color_continuous_scale='Viridis',
        opacity=0.5,
    )

    fig.update_layout(
        mapbox_style='carto-darkmatter',
        mapbox_zoom=3,
        mapbox_center={'lat': coords[0], 'lon': coords[1]},
    )  # Spatial chart specific layout options.

    fig = update_fig_layout(fig)

    return fig


# Plot a time series chart
def plot_timeseries(df) -> go.Figure:
    time = df['timeChar']
    fig = px.line(df, x=time, y="value", color='variables')

    fig = update_fig_layout(fig)

    return fig


# Plot a control chart
def plot_control(df, show_avg, show_deviations, deviation_coefficient, show_trends) -> go.Figure:
    # Additional processing necessary for control chart plotting.
    df_plot, segments = sort_cube_data(df, show_avg='all', show_deviations='all', show_trends='updown')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['time'],
        y=df_plot['value'],
        mode='lines',
        line=dict(color=FLAGS['base'][1]),
        showlegend=False,
    ))  # Base line plot

    for key in FLAGS.keys():
        summ_df = df_plot[df_plot.flag == key]
        if show_avg != 'none':
            if key == 'above_avg' and show_avg != 'below':
                marker_name = FLAGS[key][0]
                marker_color = FLAGS[key][1]
                fig = add_markers(summ_df, fig, marker_name, marker_color)
            if key == 'below_avg' and show_avg != 'above':
                marker_name = FLAGS[key][0]
                marker_color = FLAGS[key][1]
                fig = add_markers(summ_df, fig, marker_name, marker_color)

        if show_deviations != 'none':
            if key == 'deviation_above' and show_avg != 'below':
                marker_name = FLAGS[key][0]
                marker_color = FLAGS[key][1]
                fig = add_markers(summ_df, fig, marker_name, marker_color)
            if key == 'deviation_below' and show_avg != 'above':
                marker_name = FLAGS[key][0]
                marker_color = FLAGS[key][1]
                fig = add_markers(summ_df, fig, marker_name, marker_color)

    if show_trends != 'none':
        for start_idx, end_idx in zip(segments[:-1], segments[1:]):
            segment = df_plot.iloc[start_idx:end_idx + 1, :].copy()

            segment['serial_time'] = [(d - datetime.datetime(1970, 1, 1)).days for d in segment['time']]

            x = sm.add_constant(segment['serial_time'])
            model = sm.OLS(segment['value'], x).fit()
            segment['fitted_values'] = model.fittedvalues

            fit_color = COLOR_STYLES['marker_colors'][4] if model.params['serial_time'] > 0 \
                else COLOR_STYLES['marker_colors'][5]

            trend_name = "Trending Up" if model.params['serial_time'] > 0 else "Trending Down"

            print_trend = False

            if show_trends == 'all':
                print_trend = True
            else:
                if model.f_pvalue < 0.05:
                    if show_trends == 'up' and model.params['serial_time'] > 0:
                        print_trend = True
                    elif show_trends == 'down' and model.params['serial_time'] <= 0:
                        print_trend = True
                    elif show_trends == 'updown':
                        print_trend = True
                    else:
                        pass
                else:
                    pass

            if print_trend:
                fig.add_trace(go.Scatter(
                    x=segment['time'],
                    y=segment['fitted_values'],
                    mode='lines',
                    line=dict(color=fit_color),
                    name=trend_name,
                ))

        # Ensure duplicate legend items get filtered
        legend_names = set()
        fig.for_each_trace(
            lambda trace:
            trace.update(showlegend=False) if (trace.name in legend_names) else legend_names.add(trace.name)
        )

    return fig


# Plot a Histogram of the chart data
def plot_histogram(df_plot, histo_type, histo_highlight, discrete_latlong_size, bin_size) -> go.Figure:
    fig = go.Figure()

    variables = list(pd.unique(df_plot['variables']))
    years = list(pd.unique(df_plot['year']))

    variable_alias = ['var', 'variables', 'variable']
    latitude_alias = ['lat', 'latitude']
    longitude_alias = ['lon', 'long', 'longitude']

    # Create splits in the data to highlight geographically.
    if histo_highlight in latitude_alias or histo_highlight in longitude_alias:
        bins, bins_labels = make_bins(discrete_latlong_size, bin_min=-90.0, bin_max=90.0)
        if histo_highlight in latitude_alias:
            df_plot['bins'] = pd.cut(x=df_plot['lat'], bins=bins, labels=bins_labels)
        elif histo_highlight in longitude_alias:
            df_plot['bins'] = pd.cut(x=df_plot['lon'], bins=bins, labels=bins_labels)

    # Singular Histogram chart output
    if histo_type == 'single':
        if histo_highlight in variable_alias:
            for variable in variables:
                trace = construct_trace(df=df_plot, filters=[variable], highlight='variable')
                trace['name'] = f"variable: {variable}"
                fig.add_trace(trace)
            fig.update_layout(barmode='stack')

        elif histo_highlight in latitude_alias or histo_highlight in longitude_alias:
            for bins in pd.unique(df_plot['bins']):
                trace = construct_trace(df=df_plot, filters=[bins], highlight='geographic')
                trace['name'] = f"{histo_highlight}: {bins}"
                fig.add_trace(trace)

            fig.update_layout(barmode='stack')

    # Histogram Chart Output with subplots.
    elif histo_type == 'multi':
        subplot_count = len(variables)
        subplot_col = 2
        subplot_row = math.ceil(subplot_count / 2)
        variable_count = 0
        fig = make_subplots(rows=subplot_row, cols=subplot_col)

        for col in range(0, subplot_col):
            for row in range(0, subplot_row):
                if variable_count <= len(variables):
                    if histo_highlight in variable_alias:
                        trace = construct_trace(df=df_plot, filters=[variables[variable_count]], highlight='variable')
                        trace['name'] = f"variable: {variables[variable_count]}"
                        fig.add_trace(trace, row=row + 1, col=col + 1)
                        fig.update_layout(barmode='stack')

                    elif histo_highlight in latitude_alias or histo_highlight in longitude_alias:
                        for bins in pd.unique(df_plot['bins']):
                            trace = construct_trace(df=df_plot,
                                                    filters=[bins, variables[variable_count]],
                                                    highlight='geographic')
                            trace['name'] = f"variable: {variables[variable_count]}, {histo_highlight}: {bins}"
                            fig.add_trace(trace, row=row + 1, col=col + 1)

                        fig.update_layout(barmode='stack')
                    else:
                        raise ValueError(f"{histo_highlight} is not a valid highlight.")
                    variable_count += 1

    # Animated Histogram Chart by Year.
    elif histo_type == 'animated':
        fig_frames = []
        max_bin = 0
        view_max = df_plot['value'].max() * 1.025
        view_min = df_plot['value'].min() * 0.975

        # Make Slider
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Year:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        # Making Frames
        for year in years:

            frame_data = []

            if histo_highlight in variable_alias:
                for variable in variables:
                    data = {
                        'type': 'histogram',
                        'x': np.array(
                            df_plot['value'].loc[(df_plot['year'] == year) & (df_plot['variables'] == variable)]),
                        'name': str(variable),
                        'showlegend': True
                    }
                    frame_data.append(data)
                    absolute_max_bin = len(data['x'])
                    max_bin = max(max_bin, absolute_max_bin)

            if histo_highlight in latitude_alias:
                for bins in pd.unique(df_plot['bins']):
                    data = {
                        'type': 'histogram',
                        'x': np.array(df_plot['value'].loc[(df_plot['year'] == year) & (df_plot['bins'] == bins)]),
                        'name': f"Latitude: {bins}",
                        'showlegend': True
                    }
                    frame_data.append(data)
                    absolute_max_bin = len(data['x'])
                    max_bin = max(max_bin, absolute_max_bin)

            if histo_highlight in longitude_alias:
                for bins in pd.unique(df_plot['bins']):
                    data = {
                        'type': 'histogram',
                        'x': np.array(df_plot['value'].loc[(df_plot['year'] == year) & (df_plot['bins'] == bins)]),
                        'name': f"Longitude: {bins}",
                        'showlegend': True
                    }
                    frame_data.append(data)
                    absolute_max_bin = len(data['x'])
                    max_bin = max(max_bin, absolute_max_bin)

            frame = go.Frame(data=frame_data, name=str(year))

            fig_frames.append(frame)

            slider_step = {"args": [
                [year],
                {"frame": {"duration": 300, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
            ],
                "label": str(year),
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        # Making the final Plot and Layout
        fig = go.Figure(
            data=fig_frames[0]['data'],
            layout=go.Layout(
                xaxis=dict(
                    range=[view_min, view_max],
                    autorange=False
                ),
                yaxis=dict(
                    range=[0, max_bin],
                    autorange=False
                ),
                title="Histogram animated",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                      method="animate",
                                      args=[None, {"frame": {"duration": 500, "redraw": True},
                                                   "fromcurrent": True,
                                                   "transition": {"duration": 300}}]
                                      ),
                                 dict(
                                     label="Pause",
                                     method="animate",
                                     args=[[None], {"frame": {"duration": 0, "redraw": True},
                                                    "mode": "immediate",
                                                    "transition": {"duration": 0}}]
                                 )],
                        showactive=False,
                    )
                ],
                sliders=[sliders_dict],
                barmode='stack'
            ),
            frames=fig_frames
        )

        rounded_bounds = (round(df_plot['value'].max(), -3) - round(df_plot['value'].min(), -3))
        bin_count = rounded_bounds / bin_size

        fig.update_traces(xbins=dict(
            start=view_min,
            end=view_max,
            size=(rounded_bounds / bin_count)
        ))

    else:
        raise ValueError(f"{histo_type} is not a valid histogram type.")

    return fig


# Make a box plot
def plot_box(df, variable) -> go.Figure:
    fig = go.Figure()

    if variable == None:
        var_opts = pd.unique(df['variables'])
    else:
        var_opts = variable

    for var in var_opts:
        fig.add_trace(go.Box(
            x=df['variables'].loc[df['variables'] == var],
            y=df['value'].loc[df['variables'] == var],
            name=f"variable: {var}",
            showlegend=True
        ))

    return fig


# Helper methods
########################################################################################################################
# Process Cube data for chart plotting
def organize_dataframe(cube, plot_type, variable, summary) -> pd.DataFrame:
    df = cube_to_dataframe(cube)
    shape_val = cube.get_shapeval()

    if shape_val == 4:
        if plot_type == "space":
            if variable is None:
                df_temp = df[df['variables'] == df['variables'][0]]
            else:
                df_temp = df[df['variables'] == variable]
        else:
            df_temp = df
    else:
        df_temp = df

    df_plot = df_temp.where(df_temp != cube.get_nodata_value())
    summ_df = pd.DataFrame

    if plot_type != 'space':
        if shape_val == 4:
            if summary == "mean":
                summ_df = df_plot.groupby(["time", "variables"]).mean().reset_index()
            if summary == "median":
                summ_df = df_plot.groupby(['time', "variables"]).median().reset_index()
            if summary == "min":
                idx = df_plot.groupby(['time', 'variables'])['value'].idxmin()
                summ_df = df_plot.loc[idx]
            if summary == "max":
                idx = df_plot.groupby(['time', 'variables'])['value'].idxmax()
                summ_df = df_plot.loc[idx]
        else:
            if summary == "mean":
                summ_df = df_plot.groupby('time').mean().reset_index()
            if summary == "median":
                summ_df = df_plot.groupby('time').median().reset_index()
            if summary == "min":
                idx = df_plot.groupby(['time'])['value'].idxmin()
                summ_df = df_plot.loc[idx]
            if summary == "max":
                idx = df_plot.groupby(['time'])['value'].idxmax()
                summ_df = df_plot.loc[idx]
    else:
        summ_df = df_plot

    summ_df.insert(loc=0, column='timeChar', value=summ_df['time'].astype(str))
    summ_df.insert(loc=0, column='year', value=pd.DatetimeIndex(summ_df['time']).year)

    return summ_df


# Create Histogram Trace
def construct_trace(df, bins=None, filters=[], highlight=''):
    trace = []
    filtered_df = pd.DataFrame(data={'col': [1, 2, 3]})

    if highlight == 'variable':
        filtered_df = df['value'].loc[df['variables'] == filters[0]]

    elif highlight == 'geographic':
        if len(filters) > 1:
            filtered_df = df.loc[(df['bins'] == filters[0]) & (df['variables'] == filters[1])]
        else:
            filtered_df = df.loc[df['bins'] == filters[0]]

    trace = go.Histogram(x=filtered_df['value'])

    return trace


# Add trace
def add_markers(df, fig, marker_name, marker_color) -> go.Figure:
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['value'],
        mode='markers',
        name=marker_name,
        marker_color=marker_color,
    ))

    return fig


# Chart Figure layout update
def update_fig_layout(fig) -> go.Figure:
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 20, "b": 0},
        plot_bgcolor='#252e3f',
        paper_bgcolor='#252e3f',
        font=dict(color='#7fafdf'),
    )

    return fig


# Make bin list for histogram
def make_bins(bin_size, bin_min, bin_max) -> Tuple[list, list]:
    bins = []
    bins_labels = []
    bin_val = bin_min
    previous_bin = bin_min
    while bin_val <= bin_max:
        bins.append(bin_val)
        if bin_val > bin_min:
            bins_labels.append(f"{previous_bin + 0.1} to {bin_val}")
        previous_bin = bin_val
        bin_val += bin_size

    if bin_val > bin_max and bin_max not in bins:
        bins.append(bin_max)
        bins_labels.append(f"{previous_bin + 0.1} to {bin_max}")

    return bins, bins_labels


# Validate the user input so that we catch invalid parameter arguments before doing a lot of work.
def validate_inputs(
        df,
        plot_type,
        variable,
        summary,
        show_avg,
        show_deviations,
        show_trends,
        histo_type,
        histo_highlight,
        deviation_coefficient,
        discrete_latlong_size,
        bin_size,
        show_plot,
) -> bool:

    # Dictionary of valid parameter arguments
    valid_args = {
        'variables': pd.unique(df['variables']),
        'plot_type': ['space', 'timeseries', 'control', 'histogram', 'box'],
        'summary': ['min', 'max', 'mean', 'median'],
        'show_avg': ['above', 'below', 'all', 'none'],
        'show_deviations': ['above', 'below', 'all', 'none'],
        'show_trends': ['up', 'down', 'updown', 'all', 'none'],
        'histo_type': ['single', 'multi', 'animated'],
        'histo_highlight': ['var', 'variable', 'variables', 'lat', 'latitude', 'lon', 'long', 'longitude'],
    }

    # Variable selection
    if variable not in valid_args['variables'] and variable is not None:
        raise ValueError(
            f"{variable} does not exist in the 'variables' field. Variables are: {valid_args['variables']}"
        )

    # Plot type selection
    if plot_type not in valid_args['plot_type']:
        raise ValueError(
            f"{plot_type} is not a valid plot type. Options are: {valid_args['plot_type']}"
        )

    # Aggregation method selection
    if summary not in valid_args['summary']:
        raise ValueError(
            f"{summary} is not a valid summary function. Options are: {valid_args['summary']}"
        )

    # Control Chart arguments
    if show_avg not in valid_args['show_avg']:
        raise ValueError(
            f"{show_avg} not a valid average highlight option. Options are: {valid_args['show_avg']}"
        )
    if show_deviations not in valid_args['show_deviations']:
        raise ValueError(
            f"{show_deviations} not a valid deviations highlight option. Options are: {valid_args['show_deviations']}"
        )
    if show_trends not in valid_args['show_trends']:
        raise ValueError(
            f"{show_trends} not a valid trends highlight option. Options are: {valid_args['show_trends']}"
        )

    # Histogram arguments
    if histo_type not in valid_args['histo_type']:
        raise ValueError(
            f"{histo_type} not a valid histogram chart type. Options are: {valid_args['histo_type']}"
        )
    if histo_highlight not in valid_args['histo_highlight']:
        raise ValueError(
            f"{histo_highlight} not a valid histogram highlight option. Options are: {valid_args['histo_highlight']}"
        )

    return True
