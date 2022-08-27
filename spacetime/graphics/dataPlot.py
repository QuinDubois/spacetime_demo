import numpy as np
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go

import scipy.stats as sp
import statsmodels.api as sm
import datetime

from spacetime.graphics.controlCubeSorter import sort_cube_data
from spacetime.operations.cubeToDataframe import cube_to_dataframe

COLOR_STYLES = {
    "chart_background": "#222831",
    "chart_grid": "#41434a",
    "tick_font": "#bbbbbb",
    "font": "#bbbbbb",
    "line_colors": [
        "#007f85"
    ],
    "marker_colors": [
        "#FAEA48",
        "#c47e0e",
        "#4951de",
        "#bd51c9",
        "#4cbf39",
        "#c95034",
    ]
}

# flags for sorting the selected data
FLAGS = {
    "base"              : COLOR_STYLES["line_colors"][0],
    "below_avg"     : COLOR_STYLES["marker_colors"][0],
    "above_avg"     : COLOR_STYLES["marker_colors"][1],
    "deviation_above"   : COLOR_STYLES["marker_colors"][2],
    "deviation_below"   : COLOR_STYLES["marker_colors"][3],
    "trending_up"       : COLOR_STYLES["marker_colors"][4],
    "trending_down"     : COLOR_STYLES["marker_colors"][5],
}


# Main cube plotting method
########################################################################################################################
def plot_cube(
        cube,
        plot_type: str = "time_series",
        variable: str = None,
        summary: str = "mean",
        showavg: str = "all",
        showdeviations: str = "all",
        deviation_coefficient=1,
        showtrends: str = "updown",
        show_plot: bool = True,
) -> None:
    df_plot = organize_dataframe(cube, plot_type, variable, summary)

    if plot_type == 'space':
        fig = plot_spatial(cube, df_plot)

    if plot_type == 'timeseries':
        fig = plot_timeseries(df_plot)

    if plot_type == 'control':
        fig = plot_control(df_plot, showavg, showdeviations, deviation_coefficient, showtrends)

    if show_plot:
        fig.show()


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
    )
    fig = update_fig_layout(fig)

    return fig


# Plot a time series chart
def plot_timeseries(df) -> go.Figure:

    time = df['timeChar']
    fig = px.line(df, x=time, y="value", color='variables')

    fig = update_fig_layout(fig)

    return fig


def plot_control(df, showavg, showdeviations, deviation_coefficient, showtrends) -> go.Figure:

    df_plot, segments = sort_cube_data(df, FLAGS, showavg='all', showdeviations='all', showtrends='updown')
    # print(df['flags'].unique())
    # print(segments)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['time'],
        y=df_plot['value'],
        mode='lines',
        line_color=COLOR_STYLES['line_colors'][0],
        showlegend=False,
    ))

    if showavg != 'none':
        summ_df = df_plot[df_plot.flag == 'above_avg']
        if not summ_df.empty:
            fig.add_trace(go.Scatter(
                x=summ_df['time'],
                y=summ_df['value'],
                mode='markers',
                name='Above Average',
                marker_color=COLOR_STYLES['marker_colors'][0],
                showlegend=True,
            ))
        summ_df = df_plot[df_plot.flag == 'below_avg']
        if not summ_df.empty:
            fig.add_trace(go.Scatter(
                x=summ_df['time'],
                y=summ_df['value'],
                mode='markers',
                name='Below Average',
                marker_color=COLOR_STYLES['marker_colors'][1],
                showlegend=True,
            ))

    if showdeviations != 'none':
        summ_df = df_plot[df_plot.flag == 'deviation_above']
        if not summ_df.empty:
            fig.add_trace(go.Scatter(
                x=summ_df['time'],
                y=summ_df['value'],
                mode='markers',
                name='Deviation Above',
                marker_color=COLOR_STYLES['marker_colors'][2],
                showlegend=True,
            ))
        summ_df = df_plot[df_plot.flag == 'deviation_below']
        if not summ_df.empty:
            fig.add_trace(go.Scatter(
                x=summ_df['time'],
                y=summ_df['value'],
                mode='markers',
                name='Deviation Below',
                marker_color=COLOR_STYLES['marker_colors'][3],
                showlegend=True,
            ))

    if showtrends != 'none':
        for start_idx, end_idx in zip(segments[:-1], segments[1:]):
            segment = df_plot.iloc[start_idx:end_idx + 1, :]

            # TODO: Figure out why this is still throwing a SettingWithCopyWarning.
            segment.loc[:, 'serial_time'] = [(d - datetime.datetime(1970, 1, 1)).days for d in segment.loc[:, 'time']]

            x = sm.add_constant(segment['serial_time'])
            model = sm.OLS(segment['value'], x).fit()
            segment.loc[:, 'fitted_values'] = model.fittedvalues

            fit_color = COLOR_STYLES['marker_colors'][4] if model.params['serial_time'] > 0 \
                else COLOR_STYLES['marker_colors'][5]

            print_trend = False

            if showtrends == 'all':
                print_trend = True
            else:
                if model.f_pvalue < 0.05:
                    if showtrends == 'up' and model.params['serial_time'] > 0:
                        print_trend = True
                    elif showtrends == 'down' and model.params['serial_time'] <= 0:
                        print_trend = True
                    elif showtrends == 'updown':
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
                ))

    return fig


# Helper methods
########################################################################################################################
def organize_dataframe(cube, plot_type, variable, summary) -> pd.DataFrame:
    df = cube_to_dataframe(cube)
    shape_val = cube.get_shapeval()

    if shape_val == 4:
        print("Filtering Variable")
        if variable is None:
            df_plot = df[df['variables'] == df['variables'][0]]
        elif variable not in df['variables'].unique():
            raise NotImplementedError()
        else:
            df_plot = df[df['variables'] == variable]
    else:
        df_plot = df

    print("Filtering NoData")
    df_plot = df_plot.where(df_plot != cube.get_nodata_value())

    if plot_type != 'space':
        print("GroupingBy summary")
        if shape_val == 4:
            if summary == "mean":
                df_plot = df_plot.groupby(['time', "variables"]).mean().reset_index()
            if summary == "median":
                df_plot = df_plot.groupby(['time', "variables"]).median().reset_index()
            if summary == "min":
                df_plot = df_plot.groupby(['time', "variables"]).min().reset_index()
            if summary == "max":
                df_plot = df_plot.groupby(['time', "variables"]).max().reset_index()
        else:
            if summary == "mean":
                df_plot = df_plot.groupby('time').mean().reset_index()
            if summary == "median":
                df_plot = df_plot.groupby('time').median().reset_index()
            if summary == "min":
                df_plot = df_plot.groupby('time').min().reset_index()
            if summary == "max":
                df_plot = df_plot.groupby('time').max().reset_index()

    df_plot.insert(loc=0, column='timeChar', value=df['time'].astype(str))

    return df_plot


def update_fig_layout(fig) -> go.Figure:
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 20, "b": 0},
        plot_bgcolor='#252e3f',
        paper_bgcolor='#252e3f',
        font=dict(color='#7fafdf'),
    )

    return fig
