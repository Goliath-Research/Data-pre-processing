# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "missingno",
#     "numpy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import missingno
    import numpy as np
    import pandas as pd

    plt.rcParams["figure.max_open_warning"] = 0

    return missingno, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing Missing Data

    ## Objectives

    - Count the present values in each column with a bar chart.
    - See which rows contain holes with a matrix, and read the sparkline beside it.
    - See whether columns are missing together with a nullity heatmap.
    - Group columns that are missing in a similar way with a dendrogram.

    ## Background

    A count of missing values says how many holes a column has. It does not say whether those holes sit in the same rows. `missingno` draws the holes. A bar chart shows how complete each column is. A matrix shows the holes row by row. A heatmap and a dendrogram show which columns are missing together.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. The lesson keeps one complete column, `symboling`, and the columns that have holes.
    - **Titanic.** Passengers, with holes in age, cabin, and the port of embarkation. The file is stored in this folder.
    - **Employees.** Employment records. `tax_file_no` is empty in every row. The file is stored in this folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the files are read from

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the Automobile file from the UCI website, and it can read `titanic.csv` and `employee_data.csv` from this folder.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks that download.

    This lesson reads GitHub copies of all three files. GitHub allows a browser page to read its files, so the copies work in both places. The UCI address and the local file names are kept in comments.
    """)
    return


@app.cell
def _():
    auto_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Introduction-to-Data-Science/main/"
        "04-Introduction%20to%20Pandas/public/imports-85.data"
    )
    course_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Data-pre-processing/master/Data-cleaning/"
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
    return auto_file, course_file, headers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile Dataset

    The file has 205 rows and 26 columns. `na_values='?'` reads question marks as missing. The table is wide, so the first rows are shown as HTML.
    """)
    return


@app.cell
def _(auto_file, headers, mo, pd):
    # Reading from the UCI website (works when Python runs on your computer):
    # dfa_full = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    dfa_full = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(dfa_full.head().to_html())
    return (dfa_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seven columns have holes. `normalized_losses` has 41. The rest of the columns are complete.
    """)
    return


@app.cell
def _(dfa_full):
    miss_val_a = dfa_full.isnull().sum()
    print(miss_val_a[miss_val_a > 0])
    return (miss_val_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The charts below use those seven columns plus `symboling`, which is complete. Keeping one complete column makes the full bar easy to see next to the short ones.
    """)
    return


@app.cell
def _(dfa_full, mo):
    dfa = dfa_full[
        [
            "symboling",
            "normalized_losses",
            "num_doors",
            "bore",
            "stroke",
            "horsepower",
            "peak_rpm",
            "price",
        ]
    ]
    mo.Html(dfa.head().to_html())
    return (dfa,)


@app.cell
def _(dfa):
    print(dfa.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `describe` summarizes the numeric columns. `num_doors` is text, so it is left out. The `count` row is the number of present values in each column. `normalized_losses` has 164 present values out of 205.
    """)
    return


@app.cell
def _(dfa, mo, np):
    mo.Html(np.round(dfa.describe(), 2).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Titanic Dataset

    The file has 891 rows and 12 columns. Age, cabin, and embarkation have holes. The other columns are complete.
    """)
    return


@app.cell
def _(course_file, mo, pd):
    # On this computer the file is also next to the notebook:
    # dft_full = pd.read_csv("titanic.csv")
    dft_full = pd.read_csv(course_file + "titanic.csv")
    mo.Html(dft_full.head().to_html())
    return (dft_full,)


@app.cell
def _(dft_full):
    miss_val_t = dft_full.isnull().sum()
    print(miss_val_t[miss_val_t > 0])
    return (miss_val_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The charts use `Survived`, which is complete, plus the three columns with holes. `describe` includes `Survived` and `Age` only. `Cabin` and `Embarked` are text.
    """)
    return


@app.cell
def _(dft_full):
    dft = dft_full[["Survived", "Age", "Cabin", "Embarked"]]
    print(dft.head())
    return (dft,)


@app.cell
def _(dft):
    print(dft.dtypes)
    return


@app.cell
def _(dft, np):
    print(np.round(dft.describe(), 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Employee Dataset

    The file has 1,000 rows and 8 columns. `tax_file_no` is missing in every row, so it cannot be used in an analysis.
    """)
    return


@app.cell
def _(course_file, mo, pd):
    # On this computer the file is also next to the notebook:
    # dfe = pd.read_csv("employee_data.csv")
    dfe = pd.read_csv(course_file + "employee_data.csv")
    mo.Html(dfe.head().to_html())
    return (dfe,)


@app.cell
def _(dfe):
    print(dfe.isnull().sum())
    return


@app.cell
def _(dfe):
    print(dfe.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `describe` keeps the numeric columns. The `count` of `tax_file_no` is 0.
    """)
    return


@app.cell
def _(dfe, np):
    print(np.round(dfe.describe(), 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    In the reduced Automobile table, which column has the most missing values? Replace `None` with your code.
    """)
    return


@app.cell
def _():
    student_most_missing = None
    print(student_most_missing)
    return (student_most_missing,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_most_missing = dfa.isnull().sum().idxmax()
```
**`normalized_losses`.** It has 41 missing values. `symboling` has none.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bar chart

    `missingno.bar` draws one bar per column. The height is the number of present values, so a short bar is a column with many holes.
    """)
    return


@app.cell
def _(dfa, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.bar(dfa, fontsize=10, ax=_axes)
    _axes.set_title("Automobile Dataset")
    return _figure


@app.cell
def _(dft, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.bar(dft, fontsize=10, ax=_axes)
    _axes.set_title("Titanic Dataset")
    return _figure


@app.cell
def _(dfe, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.bar(dfe, fontsize=10, ax=_axes)
    _axes.set_title("Employee Dataset")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `color` sets the bar color. `sort='ascending'` puts the least complete column on the left.
    """)
    return


@app.cell
def _(dfa, dfe, dft, missingno, plt):
    _figure, _axes = plt.subplots(1, 3, figsize=(15, 4))
    missingno.bar(dfa, color="tomato", sort="ascending", fontsize=8, ax=_axes[0])
    _axes[0].set_title("Automobile Dataset")
    missingno.bar(dft, color="limegreen", sort="ascending", fontsize=8, ax=_axes[1])
    _axes[1].set_title("Titanic Dataset")
    missingno.bar(dfe, color="deepskyblue", sort="ascending", fontsize=8, ax=_axes[2])
    _axes[2].set_title("Employee Dataset")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix

    `missingno.matrix` draws each present value as a mark and each hole as a gap. The sparkline on the right runs from 0 present columns at the left to every column present at the right. A row whose sparkline is at the far right is complete. A row with more holes moves left.
    """)
    return


@app.cell
def _(dfa, missingno):
    # missingno draws the sparkline only when it creates the figure itself.
    _axes = missingno.matrix(dfa, figsize=(12, 5), fontsize=10)
    _axes.set_title("Automobile Dataset")
    return _axes.figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same matrix can take a color, written as red, green, and blue fractions from 0 to 1.
    """)
    return


@app.cell
def _(dfa, missingno):
    _axes = missingno.matrix(
        dfa, figsize=(12, 5), fontsize=10, color=(1, 99 / 255, 71 / 255)
    )
    _axes.set_title("Automobile Dataset")
    return _axes.figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reduced Automobile table has 8 columns. The last row has an 8 on the sparkline: every column is present. The first row is one step to the left: one column is missing.
    """)
    return


@app.cell
def _(dfa, mo):
    mo.Html(dfa.tail(1).to_html())
    return


@app.cell
def _(dfa, mo):
    mo.Html(dfa.head(1).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The first Automobile row has one hole. Replace `None` with the name of that column.
    """)
    return


@app.cell
def _():
    student_first_hole = None
    print(student_first_hole)
    return (student_first_hole,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_first_hole = dfa.columns[dfa.iloc[0].isnull()].tolist()
```
**`normalized_losses`.** The other seven columns in that row are present.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `sparkline=False` removes the line on the right. The marks and the gaps stay.
    """)
    return


@app.cell
def _(dfa, missingno):
    _axes = missingno.matrix(
        dfa,
        figsize=(12, 5),
        fontsize=10,
        color=(1, 99 / 255, 71 / 255),
        sparkline=False,
    )
    _axes.set_title("Automobile Dataset")
    return _axes.figure


@app.cell
def _(dft, missingno):
    _axes = missingno.matrix(
        dft, figsize=(12, 5), fontsize=10, color=(50 / 255, 205 / 255, 50 / 255)
    )
    _axes.set_title("Titanic Dataset")
    return _axes.figure


@app.cell
def _(dfe, missingno):
    _axes = missingno.matrix(
        dfe, figsize=(12, 5), fontsize=10, color=(0, 191 / 255, 1)
    )
    _axes.set_title("Employee Dataset")
    return _axes.figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One employee row has a single present value, the employee number. The sparkline for that row sits at the far left. `tax_file_no` is a gap in every row.
    """)
    return


@app.cell
def _(dfe):
    one_present = dfe[dfe.notnull().sum(axis=1) == 1]
    print(one_present)
    return (one_present,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Drop columns that are missing in every row. Replace `None` with your code. Which employee column is removed?
    """)
    return


@app.cell
def _():
    student_empty_column = None
    print(student_empty_column)
    return (student_empty_column,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_empty_column = dfe.dropna(axis=1, how="all")
```
**`tax_file_no`.** Every row is missing it. `how='all'` removes that column and keeps the columns that still have some values.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heatmap

    `missingno.heatmap` colors the correlation of the holes, not the correlation of the values.

    - A value near **+1** means the two columns are missing in the same rows.
    - A value near **-1** means a hole in one column lines up with a present value in the other.
    - A value near **0** means the holes in the two columns do not line up.
    """)
    return


@app.cell
def _(dfa, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(9, 5))
    missingno.heatmap(dfa, fontsize=12, ax=_axes)
    _axes.set_title("Automobile Dataset")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bore` and `stroke` each have 4 missing values, and those 4 rows are the same rows. Their nullity correlation is 1.
    """)
    return


@app.cell
def _(dfa):
    print("Missing bore   =", int(dfa.bore.isnull().sum()))
    print("Missing stroke =", int(dfa.stroke.isnull().sum()))
    return


@app.cell
def _(dfa, mo):
    mo.Html(dfa[dfa.stroke.isna()].to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `horsepower` and `peak_rpm` each have 2 missing values, again in the same rows, so their nullity correlation is also 1. `normalized_losses` is missing in those rows too, and it is also missing in many other rows. The extra holes keep its correlation with `horsepower` below 1.
    """)
    return


@app.cell
def _(dfa):
    print("Missing horsepower =", int(dfa.horsepower.isnull().sum()))
    print("Missing peak_rpm   =", int(dfa.peak_rpm.isnull().sum()))
    return


@app.cell
def _(dfa, mo):
    mo.Html(dfa[dfa.horsepower.isna()].to_html())
    return


@app.cell
def _(dft, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(9, 5))
    missingno.heatmap(dft, cmap="RdYlGn", fontsize=12, ax=_axes)
    _axes.set_title("Titanic Dataset")
    return _figure


@app.cell
def _(dfe, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(9, 5))
    missingno.heatmap(dfe, cmap="mako", fontsize=12, ax=_axes)
    _axes.set_title("Employee Dataset")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dendrogram

    A dendrogram joins columns that are missing in a similar pattern. Columns that meet low in the tree are missing together. Columns that meet high in the tree have different patterns of holes. `method` chooses how the distance between groups is measured. `orientation` turns the tree.
    """)
    return


@app.cell
def _(dfa, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(dfa, fontsize=12, ax=_axes)
    _axes.set_title("Automobile Dataset")
    return _figure


@app.cell
def _(dfa, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(
        dfa, method="ward", orientation="top", fontsize=12, ax=_axes
    )
    _axes.set_title("Automobile Dataset, ward, top")
    return _figure


@app.cell
def _(dfa, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(
        dfa, method="average", orientation="right", fontsize=12, ax=_axes
    )
    _axes.set_title("Automobile Dataset, average, right")
    return _figure


@app.cell
def _(dft, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(
        dft, method="centroid", orientation="top", fontsize=12, ax=_axes
    )
    _axes.set_title("Titanic Dataset, centroid, top")
    return _figure


@app.cell
def _(dfe, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(
        dfe, method="ward", orientation="top", fontsize=12, ax=_axes
    )
    _axes.set_title("Employee Dataset, ward, top")
    return _figure


@app.cell
def _(dfe, missingno, plt):
    _figure, _axes = plt.subplots(figsize=(12, 5))
    missingno.dendrogram(
        dfe, method="median", orientation="left", fontsize=12, ax=_axes
    )
    _axes.set_title("Employee Dataset, median, left")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A bar chart counts present values. A short bar is a column with many holes. An empty column, such as `tax_file_no`, has no bar to use in an analysis.
    - A matrix shows the holes row by row. The sparkline is at the right when the row is complete and moves left as holes are added. `sparkline=False` removes that line.
    - A heatmap correlates the holes. `bore` and `stroke` are missing in the same four rows, so their correlation is 1. The same is true of `horsepower` and `peak_rpm`.
    - A value near -1 means the holes in one column line up with present values in the other. A value near 0 means the holes do not line up.
    - A dendrogram groups columns with a similar pattern of holes. `method` and `orientation` change the drawing of that tree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the height of a bar from `missingno.bar` measure?
    2. In `describe`, what does the `count` row measure?
    3. What does a sparkline at the far right of a matrix mean?
    4. What does a nullity correlation near +1 mean?
    5. What does a nullity correlation near -1 mean?
    6. Why is the nullity correlation of `bore` and `stroke` equal to 1?
    7. What does a dendrogram put close together?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Present values:** The bar counts values that are there. A short bar means many holes.
2. **Present values in that column:** `count` is a row of the summary, one number per numeric column.
3. **The row is complete:** The line sits at the right when every column in that row has a value.
4. **The same rows:** Holes in one column line up with holes in the other.
5. **Opposite rows:** A hole in one column lines up with a present value in the other.
6. **The same four rows:** Each column has four holes, and they fall on the same cars.
7. **Columns with a similar pattern of holes:** Columns that are missing together meet lower in the tree.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bilogur, A. (2018). Missingno: a missing data visualization suite. *Journal of Open Source Software*, 3(22), 547. https://doi.org/10.21105/joss.00547
    - VanderPlas, J. (2017). *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media. Chapter 3.
    """)
    return


if __name__ == "__main__":
    app.run()
