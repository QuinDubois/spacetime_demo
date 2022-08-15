# Sorting methods for control chart output.
import pandas
import pandas as pd
import numpy as np


# Main Method
########################################################################################################################
def sort_cube_data(
        df: pandas.DataFrame,
        FLAGS,
        showavg="all",
        showdeviations="all",
        showtrends="updown",

) -> pandas.DataFrame:
    print("Sorting through data in cube.")

    df_sorted = df
    df_sorted.insert(loc=0, name='flags', value='base')

    if showavg != 'none':
        df_sorted = sort_average(df)
    if showdeviations != 'none':
        df_sorted = sort_deviations(df)
    if showtrends != 'none':
        df_sorted = sort_trends(df)
    else:
        df_sorted = df

    return df_sorted


# Helper methods
########################################################################################################################
def sort_average(df, flags) -> pandas.DataFrame:

    # TODO: Don't forget to figure out whether you actually need to give this method the flags.
    # TODO: Continue Here

    return df


def sort_deviations(df) -> pandas.DataFrame:
    return df


def sort_trends(df) -> pandas.DataFrame:
    return df
