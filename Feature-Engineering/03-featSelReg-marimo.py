# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "pandas",
#     "scikit-learn",
#     "scipy",
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
    # Filter-Based Feature Selection Methods (Regression)

    ## Objectives

    - Prepare the numeric Automobile columns, with `price` as the target.
    - Read the correlation of each numeric column with `price`, and the p-value of that correlation.
    - Compare `SelectFwe`, `SelectFpr`, `SelectPercentile`, and `SelectKBest`.
    - See that `f_regression` and mutual information can keep different columns.

    ## Background

    A filter scores each predictor against the target and keeps the high scores. It does not fit the regression that will later use those columns. In this lesson the target is `price`. The score is either `f_regression`, from the Pearson correlation, or `mutual_info_regression`.

    The lesson does not train a price model and does not measure its accuracy. It shows which columns each filter keeps.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. Rows with a missing value are removed. `price` is the target. The other numeric columns are the predictors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile Dataset

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a copy stored on GitHub. The original UCI address is kept in a comment. The table has 26 columns, so the first rows are shown as HTML.
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
    # df_raw = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    df_raw = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(df_raw.head().to_html())
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `dropna` removes every row that still has a missing value. 159 cars remain. The numeric columns are the ones with dtype `int64` or `float64`. There are 16 of them, including `price`. The predictors are the other 15. The first three predictor rows are shown as HTML.
    """)
    return


@app.cell
def _(df_raw):
    df = df_raw.dropna()
    print("Rows:", len(df))
    print("Missing values:", int(df.isnull().sum().sum()))
    return (df,)


@app.cell
def _(df):
    numeric_columns = [
        column
        for column in df.columns
        if df[column].dtype == "float64" or df[column].dtype == "int64"
    ]
    print(numeric_columns)
    return (numeric_columns,)


@app.cell
def _(df, mo, numeric_columns):
    Xnum = df[numeric_columns].drop(columns="price")
    y = df.price
    mo.Html(Xnum.head(3).to_html())
    return Xnum, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation with price

    The heatmap includes `price`, so the last row is the correlation of every numeric column with price. The cell below keeps only that row.
    """)
    return


@app.cell
def _(df, numeric_columns, plt, sns):
    _figure, _axes = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        linewidth=0.5,
        ax=_axes,
    )
    _axes.set_title("Correlation of the numeric columns")
    return _figure


@app.cell
def _(df, numeric_columns, plt, sns):
    price_correlation = df[numeric_columns].corr().tail(1)
    _figure, _axes = plt.subplots(figsize=(12, 2))
    sns.heatmap(
        price_correlation,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        linewidth=0.5,
        ax=_axes,
    )
    _axes.set_title("Correlation with price")
    return _figure, price_correlation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pearsonr` also returns a p-value. A small p-value means the correlation would be unusual if the column and `price` were unrelated. Every one of these 15 p-values is below 0.05. That does not mean a later filter will keep all 15. A filter that tests 15 columns at once uses a stricter cutoff.
    """)
    return


@app.cell
def _():
    from scipy.stats import pearsonr

    return (pearsonr,)


@app.cell
def _(Xnum, pd, pearsonr, plt, sns, y):
    p_values = [pearsonr(Xnum[column], y)[1] for column in Xnum.columns]
    p_value_frame = pd.DataFrame([p_values], columns=Xnum.columns, index=["price"])
    _figure, _axes = plt.subplots(figsize=(12, 2))
    sns.heatmap(
        p_value_frame,
        annot=True,
        fmt=".4f",
        cmap="Greens",
        linewidth=0.5,
        ax=_axes,
    )
    _axes.set_title("p-value of the correlation with price")
    return _figure, p_value_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The filters

    Each filter below uses a score function and a rule for how many columns to keep.

    - `SelectFwe` uses a family-wise error rate. The cutoff is stricter because many columns are tested together.
    - `SelectFpr` keeps a column when its own p-value is below `alpha`.
    - `SelectPercentile` keeps a percentage of the highest scores.
    - `SelectKBest` keeps the `k` highest scores.

    `f_regression` turns the Pearson correlation into that score.
    """)
    return


@app.cell
def _():
    from sklearn.feature_selection import (
        SelectFpr,
        SelectFwe,
        SelectKBest,
        SelectPercentile,
        f_regression,
    )

    return SelectFpr, SelectFwe, SelectKBest, SelectPercentile, f_regression


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectFwe

    The default `alpha` is 0.05. The family-wise rule keeps 10 columns. It drops `symboling`, `normalized_losses`, `stroke`, `compression_ratio`, and `peak_rpm`, even though each raw p-value was below 0.05.
    """)
    return


@app.cell
def _(SelectFwe, Xnum, f_regression, y):
    selected_fwe = SelectFwe(score_func=f_regression, alpha=0.05)
    selected_fwe.fit(Xnum, y)
    print(list(selected_fwe.get_feature_names_out()))
    return (selected_fwe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `alpha=0.01` is stricter. The same 10 columns are tested again, and `height` is dropped as well. Nine columns remain.
    """)
    return


@app.cell
def _(SelectFwe, Xnum, f_regression, y):
    selected_fwe_strict = SelectFwe(score_func=f_regression, alpha=0.01)
    selected_fwe_strict.fit(Xnum, y)
    print(list(selected_fwe_strict.get_feature_names_out()))
    return (selected_fwe_strict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    How many columns does `SelectFwe` keep when `alpha` is 0.05? Replace `None` with your code.
    """)
    return


@app.cell
def _():
    student_fwe_count = None
    print(student_fwe_count)
    return (student_fwe_count,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_fwe_count = len(selected_fwe.get_feature_names_out())
```
**10.** `alpha=0.01` keeps 9, because `height` is dropped.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectFpr

    `SelectFpr` compares each p-value with `alpha` on its own. With `alpha=0.01` it keeps 11 columns. `height` and `compression_ratio` stay. `SelectFwe` had dropped both of those at the same alpha, because the family-wise rule is stricter.

    The kept columns are the ones whose p-values fall below `alpha`, not above it.
    """)
    return


@app.cell
def _(SelectFpr, Xnum, f_regression, y):
    selected_fpr = SelectFpr(score_func=f_regression, alpha=0.01)
    selected_fpr.fit(Xnum, y)
    print(list(selected_fpr.get_feature_names_out()))
    return (selected_fpr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectPercentile

    `percentile=10` keeps the highest tenth of the scores. Fifteen predictors make that two columns: `width` and `curb_weight`. `percentile=50` keeps the highest half. Here that is 7 columns.
    """)
    return


@app.cell
def _(SelectPercentile, Xnum, f_regression, y):
    selected_p10 = SelectPercentile(score_func=f_regression, percentile=10)
    selected_p10.fit(Xnum, y)
    print(list(selected_p10.get_feature_names_out()))
    return (selected_p10,)


@app.cell
def _(SelectPercentile, Xnum, f_regression, y):
    selected_p50 = SelectPercentile(score_func=f_regression, percentile=50)
    selected_p50.fit(Xnum, y)
    print(list(selected_p50.get_feature_names_out()))
    return (selected_p50,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectKBest and f_regression

    `k=8` keeps eight columns. `k=4` keeps `length`, `width`, `curb_weight`, and `engine_size`. The names come out in their original order, not in score order.
    """)
    return


@app.cell
def _(SelectKBest, Xnum, f_regression, y):
    selected_k8 = SelectKBest(score_func=f_regression, k=8)
    selected_k8.fit(Xnum, y)
    print(list(selected_k8.get_feature_names_out()))
    return (selected_k8,)


@app.cell
def _(SelectKBest, Xnum, f_regression, y):
    selected_k4 = SelectKBest(score_func=f_regression, k=4)
    selected_k4.fit(Xnum, y)
    print(list(selected_k4.get_feature_names_out()))
    return (selected_k4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectKBest and mutual information

    Mutual information can see a relationship that is not a straight line. The calculation samples the rows, so this lesson fixes `random_state=0` and the same call returns the same columns.

    With that seed, `k=8` matches the eight columns from `f_regression`. `k=4` does not. Mutual information keeps `curb_weight`, `engine_size`, `horsepower`, and `highway_mpg`.
    """)
    return


@app.cell
def _():
    from functools import partial

    from sklearn.feature_selection import mutual_info_regression

    mutual_info = partial(mutual_info_regression, random_state=0)
    return (mutual_info,)


@app.cell
def _(SelectKBest, Xnum, mutual_info, y):
    selected_mi8 = SelectKBest(score_func=mutual_info, k=8)
    selected_mi8.fit(Xnum, y)
    print(list(selected_mi8.get_feature_names_out()))
    return (selected_mi8,)


@app.cell
def _(SelectKBest, Xnum, mutual_info, y):
    selected_mi4 = SelectKBest(score_func=mutual_info, k=4)
    selected_mi4.fit(Xnum, y)
    print(list(selected_mi4.get_feature_names_out()))
    return (selected_mi4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Keep the 4 best columns with `f_regression`. Replace `None` with your code. Is `horsepower` in the result?
    """)
    return


@app.cell
def _():
    student_k4 = None
    print(student_k4)
    return (student_k4,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_k4 = SelectKBest(score_func=f_regression, k=4).fit(Xnum, y)
print(list(student_k4.get_feature_names_out()))
```
**No.** The four columns are `length`, `width`, `curb_weight`, and `engine_size`. `horsepower` is one of the four kept by mutual information, not by `f_regression`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - After dropping incomplete rows, 159 cars and 15 numeric predictors remain. `price` is the target.
    - Every predictor has a Pearson p-value below 0.05. `SelectFwe` still drops five of them at `alpha=0.05`, because it corrects for testing 15 columns. At `alpha=0.01` it also drops `height`.
    - `SelectFpr` keeps a column whose own p-value is below `alpha`. At 0.01 it keeps `height` and `compression_ratio`, which `SelectFwe` drops.
    - `SelectPercentile` with `percentile=10` keeps `width` and `curb_weight`. `percentile=50` keeps 7 columns.
    - `SelectKBest` with `f_regression` and `k=4` keeps `length`, `width`, `curb_weight`, and `engine_size`. Mutual information with the same `k`, and with `random_state=0`, keeps `curb_weight`, `engine_size`, `horsepower`, and `highway_mpg`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Does this lesson fit a regression model and report its accuracy?
    2. What is the target column?
    3. Why can `SelectFwe` drop a column whose p-value is below 0.05?
    4. Which extra column does `alpha=0.01` remove from the `SelectFwe` result?
    5. Does `SelectFpr` keep columns with p-values above `alpha`, or below it?
    6. Which two columns does `percentile=10` keep?
    7. Do `f_regression` and mutual information keep the same four columns when `k=4`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **No:** The filters only score columns against `price`. They do not train a price model.
2. **`price`:** The other numeric columns are the predictors.
3. **It tests many columns:** The family-wise rule uses a stricter cutoff than the raw p-value.
4. **`height`:** The 0.05 result includes it. The 0.01 result does not.
5. **Below it:** A column stays when its p-value is smaller than `alpha`.
6. **`width` and `curb_weight`:** Those are the highest tenth of the `f_regression` scores.
7. **No:** `f_regression` keeps `length` and `width`. Mutual information keeps `horsepower` and `highway_mpg`. Both keep `curb_weight` and `engine_size`.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [sklearn.feature_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
    """)
    return


if __name__ == "__main__":
    app.run()
