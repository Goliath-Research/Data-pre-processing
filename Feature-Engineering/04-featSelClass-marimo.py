# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams["figure.max_open_warning"] = 0

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Filter-Based Feature Selection Methods (Classification)

    ## Objectives

    - Read the Car Evaluation categories and see how often each value appears.
    - Test whether each category is associated with `car_class`, using a chi-square test.
    - Encode the ordered categories as integers, then compare `SelectFwe`, `SelectFpr`, `SelectPercentile`, and `SelectKBest`.

    ## Background

    A filter scores each predictor against the class and keeps the high scores. It does not train a classifier. The Car Evaluation file is entirely categorical. A chi-square test can use those categories directly. The scikit-learn filters in this lesson need non-negative numbers, so the ordered categories are mapped to integers first.

    The lesson does not measure the accuracy of a classifier.

    ## Datasets Used

    - **Car Evaluation Dataset** from the UCI Machine Learning Repository. Six categorical predictors, and `car_class` as the target. There are 1,728 rows and no missing values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Car Evaluation

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website, and it can read `car.data` from this folder.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a GitHub copy. The UCI address and the local file name are kept in comments. The table has 7 columns, so the first rows are shown as HTML.
    """)
    return


@app.cell
def _():
    car_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Data-pre-processing/master/Feature-Engineering/car.data"
    )
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_class"]
    return car_file, headers


@app.cell
def _(car_file, headers, mo, pd):
    # Reading from the UCI website (works when Python runs on your computer):
    # df = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    #     header=None,
    #     names=headers,
    # )
    # On this computer the file is also next to the notebook:
    # df = pd.read_csv("car.data", header=None, names=headers)
    df = pd.read_csv(car_file, header=None, names=headers)
    mo.Html(df.head().to_html())
    return (df,)


@app.cell
def _(df):
    print(df.isnull().sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each predictor is balanced: every value appears the same number of times. `car_class` is not: most cars are unacceptable.
    """)
    return


@app.cell
def _(df, plt):
    predictor_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    _figure, _axes = plt.subplots(2, 3, figsize=(12, 7))
    for _axis, _name in zip(_axes.ravel(), predictor_names):
        df[_name].value_counts().plot(kind="bar", rot=0, ax=_axis, title=_name)
    _figure.tight_layout()
    _figure
    return _figure, predictor_names


@app.cell
def _(df, plt):
    _figure, _axes = plt.subplots(figsize=(6, 4))
    df.car_class.value_counts().plot(kind="bar", rot=0, ax=_axes, title="car_class")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chi-square test of independence

    The test asks whether the class mix is the same for every value of a predictor. A small p-value means the counts would be unusual if the predictor and `car_class` were independent. This lesson uses 0.05 as the cutoff.

    `doors` is the only predictor whose p-value is above 0.05. The others round to 0 at six decimal places.
    """)
    return


@app.cell
def _():
    from scipy.stats import chi2_contingency

    return (chi2_contingency,)


@app.cell
def _(chi2_contingency, df, pd, predictor_names):
    chi_rows = []
    for _name in predictor_names:
        _table = pd.crosstab(df[_name], df.car_class)
        chi_rows.append(
            {"feature": _name, "p_value": chi2_contingency(_table)[1]}
        )
    chi_pvalues = pd.DataFrame(chi_rows)
    print(chi_pvalues.round(6))
    return (chi_pvalues,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The buying-price chart splits the class counts by price level. Its p-value is below 0.05, so the class mix changes with the buying price.
    """)
    return


@app.cell
def _(chi2_contingency, df, pd, plt):
    buying_table = pd.crosstab(df.buying, df.car_class)
    buying_p = chi2_contingency(buying_table)[1]
    _figure, _axes = plt.subplots(figsize=(8, 5))
    buying_table.plot(kind="bar", rot=0, ax=_axes, title=f"buying and car_class, p = {buying_p:.6f}")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The doors chart tells a different story. Its p-value is about 0.3202, which is above 0.05. The class mix does not change enough, from two doors to five or more, for this test to call the columns associated.
    """)
    return


@app.cell
def _(chi2_contingency, df, pd, plt):
    doors_table = pd.crosstab(df.doors, df.car_class)
    doors_p = chi2_contingency(doors_table)[1]
    _figure, _axes = plt.subplots(figsize=(8, 5))
    doors_table.plot(kind="bar", rot=0, ax=_axes, title=f"doors and car_class, p = {doors_p:.4f}")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Integer codes

    The filters below use `chi2` from scikit-learn, which needs non-negative numbers. The dictionaries follow the order of each category. They are ranks, not the original words.

    `persons` is the easy one to misread. Two people are coded as 1, four people as 2, and more than four as 3. The code 2 does not mean two people.

    `car_class` is coded only so it can be the target: unacceptable 1, acceptable 2, good 3, very good 4.
    """)
    return


@app.cell
def _(df, mo):
    buying_dict = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
    coded = df.copy()
    coded["buying"] = coded["buying"].map(buying_dict)
    coded["maint"] = coded["maint"].map(buying_dict)
    coded["doors"] = coded["doors"].map({"2": 2, "3": 3, "4": 4, "5more": 5})
    coded["persons"] = coded["persons"].map({"2": 1, "4": 2, "more": 3})
    coded["lug_boot"] = coded["lug_boot"].map({"small": 1, "med": 2, "big": 3})
    coded["safety"] = coded["safety"].map({"low": 1, "med": 2, "high": 3})
    coded["car_class"] = coded["car_class"].map(
        {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}
    )
    mo.Html(coded.head().to_html())
    return buying_dict, coded


@app.cell
def _(coded):
    X = coded[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]
    y = coded.car_class
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What integer does the persons code give a car with 4 people? Replace `None` with that integer.
    """)
    return


@app.cell
def _():
    student_four_people = None
    print(student_four_people)
    return (student_four_people,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_four_people = 2
```
**2.** The dictionary is `{"2": 1, "4": 2, "more": 3}`. Four people are the middle rank, not the number 4.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SelectFwe and SelectFpr

    `SelectFwe` uses a family-wise cutoff. `SelectFpr` uses each column's own p-value. With chi-square, `SelectFwe` at `alpha=0.05` and `SelectFpr` at `alpha=0.001` both drop `doors` and keep the other five predictors. That matches the contingency test, where `doors` was the column whose p-value was above 0.05.

    Five columns are shown as HTML.
    """)
    return


@app.cell
def _():
    from sklearn.feature_selection import (
        SelectFpr,
        SelectFwe,
        SelectKBest,
        SelectPercentile,
        chi2,
    )

    return SelectFpr, SelectFwe, SelectKBest, SelectPercentile, chi2


@app.cell
def _(SelectFwe, X, chi2, mo, pd, y):
    selected_fwe = SelectFwe(score_func=chi2, alpha=0.05)
    selected_fwe.fit(X, y)
    kept_fwe = pd.DataFrame(
        selected_fwe.transform(X), columns=selected_fwe.get_feature_names_out()
    )
    mo.Html(kept_fwe.head(3).to_html())
    return kept_fwe, selected_fwe


@app.cell
def _(SelectFpr, X, chi2, mo, pd, y):
    selected_fpr = SelectFpr(score_func=chi2, alpha=0.001)
    selected_fpr.fit(X, y)
    kept_fpr = pd.DataFrame(
        selected_fpr.transform(X), columns=selected_fpr.get_feature_names_out()
    )
    mo.Html(kept_fpr.head(3).to_html())
    return kept_fpr, selected_fpr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SelectPercentile and SelectKBest

    `percentile=30` keeps about 30 percent of the six columns, which is two columns. Chi-square and mutual information, with `random_state=0`, both keep `persons` and `safety`.

    `SelectKBest` with `k=2` keeps that same pair. `k=4` with chi-square adds `buying` and `maint`. `lug_boot` has a tiny contingency p-value, and it is still outside this top four.
    """)
    return


@app.cell
def _(SelectPercentile, X, chi2, y):
    selected_p30 = SelectPercentile(score_func=chi2, percentile=30)
    selected_p30.fit(X, y)
    print(list(selected_p30.get_feature_names_out()))
    return (selected_p30,)


@app.cell
def _():
    from functools import partial

    from sklearn.feature_selection import mutual_info_classif

    mutual_info = partial(mutual_info_classif, random_state=0)
    return (mutual_info,)


@app.cell
def _(SelectPercentile, X, mutual_info, y):
    selected_p30_mi = SelectPercentile(score_func=mutual_info, percentile=30)
    selected_p30_mi.fit(X, y)
    print(list(selected_p30_mi.get_feature_names_out()))
    return (selected_p30_mi,)


@app.cell
def _(SelectKBest, X, chi2, y):
    selected_k2 = SelectKBest(score_func=chi2, k=2)
    selected_k2.fit(X, y)
    print(list(selected_k2.get_feature_names_out()))
    return (selected_k2,)


@app.cell
def _(SelectKBest, X, chi2, y):
    selected_k4 = SelectKBest(score_func=chi2, k=4)
    selected_k4.fit(X, y)
    print(list(selected_k4.get_feature_names_out()))
    return (selected_k4,)


@app.cell
def _(SelectKBest, X, mutual_info, y):
    selected_mi2 = SelectKBest(score_func=mutual_info, k=2)
    selected_mi2.fit(X, y)
    print(list(selected_mi2.get_feature_names_out()))
    return (selected_mi2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which predictor is missing from the `SelectFwe` result? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_dropped = None
    print(student_dropped)
    return (student_dropped,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_dropped = "doors"
```
**`doors`.** It is the predictor whose contingency p-value is about 0.3202. The other five predictors are kept.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - The file has six categorical predictors and no missing values. `car_class` is mostly `unacc`.
    - A chi-square test of independence gives `doors` a p-value of about 0.3202. The other five predictors have p-values that round to 0.
    - The integer codes follow each category's order. The persons code 2 means four people, not two.
    - `SelectFwe` at `alpha=0.05` and `SelectFpr` at `alpha=0.001` both drop `doors` and keep the other five predictors.
    - `percentile=30` and `k=2` keep `persons` and `safety`, for chi-square and for mutual information with `random_state=0`.
    - Chi-square with `k=4` keeps `buying`, `maint`, `persons`, and `safety`. `lug_boot` is associated with the class and is still outside that top four.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Does this lesson report the accuracy of a classifier?
    2. Which predictor has a chi-square p-value above 0.05?
    3. What does a small p-value say about a predictor and `car_class`?
    4. What does the persons code 2 mean?
    5. Which predictor do `SelectFwe` and `SelectFpr` drop?
    6. Which two predictors do `k=2` and `percentile=30` keep?
    7. Is `lug_boot` in the chi-square top four?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **No:** The filters score predictors against the class. They do not train a classifier.
2. **`doors`:** Its p-value is about 0.3202. The other five round to 0.
3. **They are associated:** The class mix would be unusual if that predictor and the class were independent.
4. **Four people:** The codes are ranks. Two people are 1, four people are 2, and more than four are 3.
5. **`doors`:** The other five predictors stay.
6. **`persons` and `safety`:** Chi-square and mutual information agree on that pair in this lesson.
7. **No:** The four are `buying`, `maint`, `persons`, and `safety`.
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
