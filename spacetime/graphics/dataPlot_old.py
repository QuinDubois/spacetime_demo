import numpy as np
import plotly_express as px


def plot_cube_old(cube, type="space", variable=None, summary="mean", showPlot=True):
    # load data
    ds = cube.get_raster_data()
    shapeVal = len(ds.shape)

    # if 3d or 4d data
    if shapeVal == 4:

        df = ds.to_dataframe(name="value", dim_order=["lat", "lon", "variables", "time"])
        df = df.reset_index()

        if type == "space":
            if variable == None:
                dfPlot = df[df["variables"] == df["variables"][0]]
            else:
                dfPlot = df[df["variables"] == variable]
        else:
            dfPlot = df

    else:

        df = ds.to_dataframe(name="value", dim_order=["lat", "lon", "time"])
        dfPlot = df.reset_index()

    print(dfPlot.dtypes)

    if type == "space":

        dfPlot = dfPlot.where(dfPlot != cube.get_nodata_value())

        dfPlot.insert(loc=0, column='timeChar', value=dfPlot["time"].astype(str))
        time = dfPlot["timeChar"]
        maxVal = np.nanmax(df["value"])

        out = dfPlot

        coords = cube.upper_left_corner()
        # print(coords)
        # plot the map
        fig = px.scatter_mapbox(dfPlot, lat='lat',
                                lon='lon',
                                color='value',
                                animation_frame=time,
                                range_color=(0, maxVal),
                                color_continuous_scale="Viridis",
                                opacity=0.5,
                                )
        fig.update_layout(mapbox_style="carto-darkmatter", mapbox_zoom=3,
                          mapbox_center={"lat": coords[0], "lon": coords[1]}, )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 20, "b": 0},
                          plot_bgcolor="#252e3f",
                          paper_bgcolor="#252e3f",
                          font=dict(color="#7fafdf"),
                          )

        if showPlot == True:
            fig.show()

    if type == "time_series":

        dfPlot = dfPlot.where(dfPlot != cube.get_nodata_value())

        if shapeVal == 4:

            if summary == "mean":
                summDF = dfPlot.groupby(['time', "variables"]).mean().reset_index()
            if summary == "median":
                summDF = dfPlot.groupby(['time', "variables"]).median().reset_index()
            if summary == "min":
                summDF = dfPlot.groupby(['time', "variables"]).min().reset_index()
            if summary == "max":
                summDF = dfPlot.groupby(['time', "variables"]).max().reset_index()

            summDF.insert(loc=0, column='timeChar', value=summDF["time"].astype(str))
            time = summDF["timeChar"]
            fig = px.line(summDF, x=time, y="value", color='variables')

        else:

            if summary == "mean":
                summDF = dfPlot.groupby('time').mean().reset_index()
            if summary == "median":
                summDF = dfPlot.groupby('time').median().reset_index()
            if summary == "min":
                summDF = dfPlot.groupby('time').min().reset_index()
            if summary == "max":
                summDF = dfPlot.groupby('time').max().reset_index()

            summDF.insert(loc=0, column='timeChar', value=summDF["time"].astype(str))
            time = summDF["timeChar"]
            fig = px.line(summDF, x=time, y="value")

        fig.update_layout(margin={"r": 0, "t": 0, "l": 20, "b": 0},
                          plot_bgcolor="#252e3f",
                          paper_bgcolor="#252e3f",
                          font=dict(color="#7fafdf"),
                          )

        if showPlot == True:
            fig.show()

        out = summDF

    return out