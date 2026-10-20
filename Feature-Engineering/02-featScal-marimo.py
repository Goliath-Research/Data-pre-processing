# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Scaling

    ## Objectives

    - See that `length`, `width`, and `height` are measured on different ranges.
    - Rescale them with min-max scaling, max-abs scaling, a z-score, and a robust scaler.
    - Compare the center and the spread that each method produces.

    ## Background

    Length, width, and height are all lengths, but their numbers are not the same size. A method that uses the raw numbers can treat the column with the larger numbers as the more important one. Scaling puts the columns on a comparable range. This lesson shows the scaled numbers and their distributions. It does not fit a model, so it does not measure a change in model accuracy.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. The scaled columns are `length`, `width`, and `height`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile dataset

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a copy stored on GitHub. GitHub allows a browser page to read its files. The original UCI address is kept in a comment.

    The full table has 26 columns, so the first rows are shown as HTML. The lesson then keeps three columns.
    """)
    return


@app.cell
def _():
    auto_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Introduction-to-Data-Science/main/"
        "04-Introduction%20to%20Pandas/public/imports-85.data"
    )
    headers = [
        "symboling",
        "normalized_losses",
        "make",
        "fuel_type",
        "aspiration",
        "num_doors",
        "body_style",
        "drive_wheels",
        "engine_location",
        "wheel_base",
        "length",
        "width",
        "height",
        "curb_weight",
        "engine_type",
        "num_cylinders",
        "engine_size",
        "fuel_system",
        "bore",
        "stroke",
        "compression_ratio",
        "horsepower",
        "peak_rpm",
        "city_mpg",
        "highway_mpg",
        "price",
    ]
    return auto_file, headers


@app.cell
def _(auto_file, headers, mo, pd):
    # df = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    df = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(df.head().to_html())
    return (df,)


@app.cell
def _(df):
    dfn = df[["length", "width", "height"]].copy()
    print(dfn.head())
    return (dfn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A boxenplot and three density curves show the raw columns on one axis. Length sits far to the right of width and height.
    """)
    return


@app.cell
def _(dfn, pd, plt, sns):
    dfn_melted = pd.melt(
        dfn, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=dfn_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Length, width, and height")
    return _figure, dfn_melted


@app.cell
def _(dfn, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn, fill=True, label="height", ax=_axes)
    _axes.set_title("Length, width, and height")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Min-max scaling

    `MinMaxScaler` maps the smallest value in a column to 0 and the largest to 1. Every other value falls between them. The default range is 0 to 1.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import MinMaxScaler

    return (MinMaxScaler,)


@app.cell
def _(MinMaxScaler, dfn, pd):
    dfn_mms = pd.DataFrame(
        MinMaxScaler().fit_transform(dfn), columns=dfn.columns
    )
    print(dfn_mms.head())
    return (dfn_mms,)


@app.cell
def _(dfn_mms, pd, plt, sns):
    _melted = pd.melt(
        dfn_mms, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Min-max scaling, 0 to 1")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each scaled column now runs from 0 to 1. The shape of each density is the shape of the original column.
    """)
    return


@app.cell
def _(dfn_mms, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn_mms, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn_mms, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn_mms, fill=True, label="height", ax=_axes)
    _axes.set_title("Min-max scaling, 0 to 1")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The cell prints the minimum and maximum of scaled length. Change `"length"` to `"height"` and run the cell again. What are the two numbers?
    """)
    return


@app.cell
def _(dfn_mms):
    column_name = "length"
    print(column_name, float(dfn_mms[column_name].min()), float(dfn_mms[column_name].max()))
    return (column_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
column_name = "height"
```
**0 and 1.** Min-max scaling gives every column those two ends. Width has the same ends.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `feature_range=(-1, 1)` uses the same rule with different ends. The smallest value becomes -1 and the largest becomes 1.
    """)
    return


@app.cell
def _(MinMaxScaler, dfn, pd):
    dfn_mms2 = pd.DataFrame(
        MinMaxScaler(feature_range=(-1, 1)).fit_transform(dfn),
        columns=dfn.columns,
    )
    print(dfn_mms2.head())
    return (dfn_mms2,)


@app.cell
def _(dfn_mms2, pd, plt, sns):
    _melted = pd.melt(
        dfn_mms2, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Min-max scaling, -1 to 1")
    return _figure


@app.cell
def _(dfn_mms2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn_mms2, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn_mms2, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn_mms2, fill=True, label="height", ax=_axes)
    _axes.set_title("Min-max scaling, -1 to 1")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max-abs scaling

    `MaxAbsScaler` divides each column by its largest absolute value. The largest magnitude becomes 1. The scaler does not subtract a center, so these positive lengths stay positive. Their minimum is not 0.

    One extreme value sets the divisor. If that extreme is a mistake, it shrinks every other value in the column.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import MaxAbsScaler

    return (MaxAbsScaler,)


@app.cell
def _(MaxAbsScaler, dfn, pd):
    dfn_mas = pd.DataFrame(
        MaxAbsScaler().fit_transform(dfn), columns=dfn.columns
    )
    print(dfn_mas.head())
    return (dfn_mas,)


@app.cell
def _(dfn_mas, pd, plt, sns):
    _melted = pd.melt(
        dfn_mas, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Max-abs scaling")
    return _figure


@app.cell
def _(dfn_mas, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn_mas, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn_mas, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn_mas, fill=True, label="height", ax=_axes)
    _axes.set_title("Max-abs scaling")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Z-score

    `StandardScaler` subtracts the column mean and divides by the column standard deviation. The scaled mean is 0 and the scaled standard deviation is 1. A value of 2 is two standard deviations above the mean.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler,)


@app.cell
def _(StandardScaler, dfn, pd):
    dfn_ss = pd.DataFrame(
        StandardScaler().fit_transform(dfn), columns=dfn.columns
    )
    print(dfn_ss.head())
    return (dfn_ss,)


@app.cell
def _(dfn_ss):
    print(dfn_ss.mean().round(6))
    print(dfn_ss.std(ddof=0).round(6))
    return


@app.cell
def _(dfn_ss, pd, plt, sns):
    _melted = pd.melt(
        dfn_ss, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Z-score scaling")
    return _figure


@app.cell
def _(dfn_ss, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn_ss, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn_ss, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn_ss, fill=True, label="height", ax=_axes)
    _axes.set_title("Z-score scaling")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Robust scaling

    `RobustScaler` subtracts the median and divides by the interquartile range, the third quartile minus the first. The scaled median is 0. A few very large or very small values do not set the center or the scale, because the median and the quartiles ignore them.

    The result is not a z-score. The standard deviation is not reset to 1.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import RobustScaler

    return (RobustScaler,)


@app.cell
def _(RobustScaler, dfn, pd):
    dfn_rs = pd.DataFrame(
        RobustScaler().fit_transform(dfn), columns=dfn.columns
    )
    print(dfn_rs.head())
    return (dfn_rs,)


@app.cell
def _(dfn_rs):
    print(dfn_rs.median())
    return


@app.cell
def _(dfn_rs, pd, plt, sns):
    _melted = pd.melt(
        dfn_rs, value_vars=["length", "width", "height"], var_name="measure"
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Robust scaling")
    return _figure


@app.cell
def _(dfn_rs, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="length", data=dfn_rs, fill=True, label="length", ax=_axes)
    sns.kdeplot(x="width", data=dfn_rs, fill=True, label="width", ax=_axes)
    sns.kdeplot(x="height", data=dfn_rs, fill=True, label="height", ax=_axes)
    _axes.set_title("Robust scaling")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Print the median of robust-scaled width. Replace `None` with your code. What value do you get, and why?
    """)
    return


@app.cell
def _():
    student_width_median = None
    print(student_width_median)
    return (student_width_median,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_width_median = dfn_rs.width.median()
```
**0.** Robust scaling subtracts the median, so the median of the scaled column is 0. The same is true of length and height.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Length, compared

    The four scaled length columns sit side by side. `describe` shows the center and the spread. Min-max runs from 0 to 1. Max-abs stays positive and ends at 1. The z-score is centered at 0 with standard deviation 1. Robust scaling is centered at 0 and is not the same column as the z-score.
    """)
    return


@app.cell
def _(dfn_mas, dfn_mms, dfn_rs, dfn_ss, pd):
    df_length = pd.concat(
        {
            "Min-Max": dfn_mms.length,
            "Max Abs": dfn_mas.length,
            "Z-Score": dfn_ss.length,
            "Robust": dfn_rs.length,
        },
        axis=1,
    )
    print(df_length.head())
    return (df_length,)


@app.cell
def _(df_length):
    print(df_length.describe().round(2))
    return


@app.cell
def _(df_length, pd, plt, sns):
    _melted = pd.melt(
        df_length,
        value_vars=["Min-Max", "Max Abs", "Z-Score", "Robust"],
        var_name="measure",
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Length")
    return _figure


@app.cell
def _(df_length, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="Min-Max", data=df_length, fill=True, label="Min-Max", ax=_axes)
    sns.kdeplot(x="Max Abs", data=df_length, fill=True, label="Max Abs", ax=_axes)
    sns.kdeplot(x="Z-Score", data=df_length, fill=True, label="Z-Score", ax=_axes)
    sns.kdeplot(x="Robust", data=df_length, fill=True, label="Robust", ax=_axes)
    _axes.set_title("Length")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Width, compared
    """)
    return


@app.cell
def _(dfn_mas, dfn_mms, dfn_rs, dfn_ss, pd):
    df_width = pd.concat(
        {
            "Min-Max": dfn_mms.width,
            "Max Abs": dfn_mas.width,
            "Z-Score": dfn_ss.width,
            "Robust": dfn_rs.width,
        },
        axis=1,
    )
    print(df_width.describe().round(2))
    return (df_width,)


@app.cell
def _(df_width, pd, plt, sns):
    _melted = pd.melt(
        df_width,
        value_vars=["Min-Max", "Max Abs", "Z-Score", "Robust"],
        var_name="measure",
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Width")
    return _figure


@app.cell
def _(df_width, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="Min-Max", data=df_width, fill=True, label="Min-Max", ax=_axes)
    sns.kdeplot(x="Max Abs", data=df_width, fill=True, label="Max Abs", ax=_axes)
    sns.kdeplot(x="Z-Score", data=df_width, fill=True, label="Z-Score", ax=_axes)
    sns.kdeplot(x="Robust", data=df_width, fill=True, label="Robust", ax=_axes)
    _axes.set_title("Width")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Height, compared
    """)
    return


@app.cell
def _(dfn_mas, dfn_mms, dfn_rs, dfn_ss, pd):
    df_height = pd.concat(
        {
            "Min-Max": dfn_mms.height,
            "Max Abs": dfn_mas.height,
            "Z-Score": dfn_ss.height,
            "Robust": dfn_rs.height,
        },
        axis=1,
    )
    print(df_height.describe().round(2))
    return (df_height,)


@app.cell
def _(df_height, pd, plt, sns):
    _melted = pd.melt(
        df_height,
        value_vars=["Min-Max", "Max Abs", "Z-Score", "Robust"],
        var_name="measure",
    )
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.boxenplot(data=_melted, x="measure", y="value", ax=_axes)
    _axes.set_title("Height")
    return _figure


@app.cell
def _(df_height, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x="Min-Max", data=df_height, fill=True, label="Min-Max", ax=_axes)
    sns.kdeplot(x="Max Abs", data=df_height, fill=True, label="Max Abs", ax=_axes)
    sns.kdeplot(x="Z-Score", data=df_height, fill=True, label="Z-Score", ax=_axes)
    sns.kdeplot(x="Robust", data=df_height, fill=True, label="Robust", ax=_axes)
    _axes.set_title("Height")
    _axes.set_xlabel(None)
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - Raw length, width, and height occupy different ranges, so a plot on one axis is dominated by length.
    - Min-max scaling sends the minimum to 0 and the maximum to 1, or to the ends of another `feature_range`. The shape of the column is unchanged.
    - Max-abs scaling divides by the largest absolute value. The largest magnitude becomes 1, and the values are not recentered. One extreme value sets that divisor.
    - A z-score subtracts the mean and divides by the standard deviation. The scaled mean is 0 and the scaled standard deviation is 1.
    - Robust scaling subtracts the median and divides by the interquartile range. The scaled median is 0. The result is not a second copy of the z-score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why can the raw length column dominate a plot that also shows width and height?
    2. What values do the minimum and maximum of a column become under the default min-max scaler?
    3. What changes when `feature_range` is `(-1, 1)`?
    4. Does max-abs scaling move the center of the column to 0?
    5. What are the mean and standard deviation after a z-score?
    6. Which two summaries does a robust scaler use?
    7. Is the robust-scaled length column the same as the z-score length column?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Its numbers are larger:** On a shared axis, length sits far to the right of width and height.
2. **0 and 1:** Every value in the column is placed between those ends.
3. **The ends become -1 and 1:** The smallest value maps to -1 and the largest maps to 1.
4. **No:** It only divides by the largest absolute value. These lengths stay positive.
5. **0 and 1:** The scaler subtracts the mean and divides by the standard deviation.
6. **The median and the interquartile range:** The center is the median. The scale is the third quartile minus the first.
7. **No:** Robust scaling uses the median and the quartiles. A z-score uses the mean and the standard deviation.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
    """)
    return


if __name__ == "__main__":
    app.run()
