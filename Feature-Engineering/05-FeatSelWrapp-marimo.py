# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "mlxtend",
#     "pandas",
#     "scikit-learn",
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

    plt.rcParams["figure.max_open_warning"] = 0

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wrapper Methods for Feature Selection

    ## Objectives

    - Build a linear regression of `price` and score a set of predictors with its in-sample R2.
    - Add predictors one at a time, and remove them one at a time, until five remain.
    - Let a floating search add and remove predictors in the same run.
    - Compare those sets with an exhaustive search of every subset of three, four, or five predictors.

    ## Background

    A wrapper fits a model for each subset and keeps the subset with the better score. Here the model is a linear regression and the score is R2. `cv=0` means the R2 is computed on the same 159 cars used to fit the line. It is an in-sample score, not a score on new cars.

    Forward selection adds one predictor at a time. Backward elimination starts with every predictor and removes one at a time. An exhaustive search tries every allowed subset. It is slower, and within the sizes it was allowed to try, it can find a subset the step-by-step searches miss.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. Rows with a missing value are removed. `price` is the target. The other 15 numeric columns are the predictors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile Dataset

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a copy stored on GitHub. The original UCI address is kept in a comment.
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
def _(auto_file, headers, pd):
    # df_raw = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    df_raw = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    df = df_raw.dropna()
    numeric_columns = [
        column
        for column in df.columns
        if df[column].dtype == "float64" or df[column].dtype == "int64"
    ]
    Xnum = df[numeric_columns].drop(columns="price")
    y = df.price
    print("Rows:", len(df))
    print("Predictors:", Xnum.shape[1])
    return Xnum, df, y


@app.cell
def _():
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.linear_model import LinearRegression

    return EFS, LinearRegression, SFS


@app.cell
def _(pd):
    def subset_table(subsets):
        rows = []
        for _n_features in sorted(subsets):
            _info = subsets[_n_features]
            rows.append(
                {
                    "n_features": _n_features,
                    "r2": round(_info["avg_score"], 4),
                    "features": ", ".join(_info["feature_names"]),
                }
            )
        return pd.DataFrame(rows)

    return (subset_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward selection

    `k_features=5` stops when five predictors are in the model. `forward=True` starts from no predictors and adds the one that raises the in-sample R2 the most. `scoring='r2'` is that score. `cv=0` uses the training rows themselves.
    """)
    return


@app.cell
def _(LinearRegression, SFS, Xnum, subset_table, y):
    forward = SFS(
        LinearRegression(),
        k_features=5,
        forward=True,
        floating=False,
        verbose=0,
        scoring="r2",
        cv=0,
    )
    forward.fit(Xnum, y)
    forward_table = subset_table(forward.subsets_)
    print(forward_table)
    return forward, forward_table


@app.cell
def _(forward_table, plt):
    _figure, _axes = plt.subplots(figsize=(6, 4))
    _axes.bar(forward_table.n_features, forward_table.r2, color="salmon")
    _axes.set_xlabel("Number of features")
    _axes.set_ylabel("In-sample R2")
    _axes.set_title("Forward selection")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first predictor is `curb_weight`, with an R2 of 0.7986. The five-predictor set is `length`, `width`, `curb_weight`, `engine_size`, and `peak_rpm`, with an R2 of 0.8357.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backward elimination

    `forward=False` starts with all 15 predictors and removes one at a time until five remain. The table is sorted by the number of predictors. The row with five predictors is the set that stays.
    """)
    return


@app.cell
def _(LinearRegression, SFS, Xnum, subset_table, y):
    backward = SFS(
        LinearRegression(),
        k_features=5,
        forward=False,
        floating=False,
        verbose=0,
        scoring="r2",
        cv=0,
    )
    backward.fit(Xnum, y)
    backward_table = subset_table(backward.subsets_)
    print(backward_table)
    return backward, backward_table


@app.cell
def _(backward_table, plt):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    _axes.bar(backward_table.n_features, backward_table.r2, color="skyblue")
    _axes.set_xlabel("Number of features")
    _axes.set_ylabel("In-sample R2")
    _axes.set_title("Backward elimination")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The five predictors that remain are `width`, `curb_weight`, `engine_size`, `bore`, and `stroke`. Their R2 is 0.8351. That set is not the forward set: `bore` and `stroke` replace `length` and `peak_rpm`. `width`, `curb_weight`, and `engine_size` are in both sets. The backward R2 is a little lower.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Floating searches

    `floating=True` lets the search undo an earlier choice. In a forward search, each added predictor can be followed by removing one that no longer helps. In a backward search, a removed predictor can be put back.

    On this table, the floating forward set matches the plain forward set, and the floating backward set matches the plain backward set.
    """)
    return


@app.cell
def _(LinearRegression, SFS, Xnum, y):
    forward_floating = SFS(
        LinearRegression(),
        k_features=5,
        forward=True,
        floating=True,
        verbose=0,
        scoring="r2",
        cv=0,
    )
    forward_floating.fit(Xnum, y)
    print(sorted(forward_floating.subsets_[5]["feature_names"]))
    print(round(forward_floating.subsets_[5]["avg_score"], 4))
    return (forward_floating,)


@app.cell
def _(LinearRegression, SFS, Xnum, y):
    backward_floating = SFS(
        LinearRegression(),
        k_features=5,
        forward=False,
        floating=True,
        verbose=0,
        scoring="r2",
        cv=0,
    )
    backward_floating.fit(Xnum, y)
    print(sorted(backward_floating.subsets_[5]["feature_names"]))
    print(round(backward_floating.subsets_[5]["avg_score"], 4))
    return (backward_floating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The forward search stops at five predictors. Which of `length` and `bore` is in that set? Replace `None` with the column name.
    """)
    return


@app.cell
def _():
    student_forward_column = None
    print(student_forward_column)
    return (student_forward_column,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_forward_column = "length"
```
**`length`.** The forward set also has `width`, `curb_weight`, `engine_size`, and `peak_rpm`. `bore` is in the backward set instead.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exhaustive search

    An exhaustive search fits a model for every subset whose size is between `min_features` and `max_features`. For 15 predictors, the number of subsets of size 3, 4, and 5 is 455 + 1,365 + 3,003 = 4,823. Each one is a linear regression. The search returns the subset with the highest in-sample R2. It does not look at subsets larger than five.
    """)
    return


@app.cell
def _():
    import math

    subset_counts = {
        3: math.comb(15, 3),
        4: math.comb(15, 4),
        5: math.comb(15, 5),
    }
    print(subset_counts)
    print("Total:", sum(subset_counts.values()))
    return (subset_counts,)


@app.cell
def _(EFS, LinearRegression, Xnum, y):
    exhaustive = EFS(
        LinearRegression(),
        min_features=3,
        max_features=5,
        scoring="r2",
        cv=0,
        print_progress=False,
    )
    exhaustive.fit(Xnum, y)
    best_score = -1.0
    best_names = ()
    for _info in exhaustive.subsets_.values():
        if _info["avg_score"] > best_score:
            best_score = _info["avg_score"]
            best_names = _info["feature_names"]
    print(sorted(best_names))
    print(round(best_score, 4))
    return best_names, best_score, exhaustive


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The best of the 4,823 subsets is the forward set: `length`, `width`, `curb_weight`, `engine_size`, and `peak_rpm`, with R2 0.8357. The backward set scores 0.8351, so the exhaustive search does not choose it. Among subsets of three, four, or five predictors, the forward set has the higher in-sample R2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    How many regressions does the exhaustive search fit? Replace `None` with the number.
    """)
    return


@app.cell
def _():
    student_subset_count = None
    print(student_subset_count)
    return (student_subset_count,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_subset_count = 455 + 1365 + 3003
```
**4,823.** That is every subset of 3, 4, or 5 predictors out of 15. Subsets of other sizes are not tried.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - `cv=0` makes the R2 an in-sample score for the 159 complete cars. It is not a score on new cars.
    - Forward selection of five predictors keeps `length`, `width`, `curb_weight`, `engine_size`, and `peak_rpm`. The R2 is 0.8357. The first predictor added is `curb_weight`.
    - Backward elimination keeps `width`, `curb_weight`, `engine_size`, `bore`, and `stroke`. The R2 is 0.8351.
    - Floating searches, which may undo an earlier step, reach those same two sets on this table.
    - The exhaustive search fits 4,823 regressions and chooses the forward set. The backward set is close and is not the best subset of size 3, 4, or 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does `cv=0` do to the R2?
    2. Which predictor does forward selection add first?
    3. Which five predictors does forward selection finish with?
    4. Which two predictors are in the backward set and not in the forward set?
    5. Did the floating searches find different five-predictor sets?
    6. How many subsets does the exhaustive search fit?
    7. Which five-predictor set has the higher in-sample R2, forward or backward?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **It uses the training rows:** The score is the fit to these 159 cars. It is not cross-validated.
2. **`curb_weight`:** Its one-predictor R2 is 0.7986.
3. **`length`, `width`, `curb_weight`, `engine_size`, and `peak_rpm`:** That set has R2 0.8357.
4. **`bore` and `stroke`:** The forward set has `length` and `peak_rpm` instead.
5. **No:** Floating forward matches forward, and floating backward matches backward.
6. **4,823:** That is every subset of 3, 4, or 5 of the 15 predictors.
7. **Forward:** 0.8357 is higher than the backward score, 0.8351. The exhaustive search chooses the forward set.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Albon, C. (2018). *Python Machine Learning Cookbook: Practical Solutions from Preprocessing to Deep Learning*. O'Reilly Media. Chapter 10.
    - [mlxtend feature selection](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/)
    """)
    return


if __name__ == "__main__":
    app.run()
