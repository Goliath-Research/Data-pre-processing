import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Removing Missing Values

    ## Objectives

    - See how `None` and `NaN` change the dtype of a NumPy array and the result of `mean`.
    - Find missing values in a DataFrame with `isnull` and `notnull`.
    - Remove rows and columns with `dropna`, using `how`, `thresh`, and `inplace`.
    - Read the Automobile file so that `?` becomes missing, and see the dtype when `?` is left as text.

    ## Background

    A missing value is a hole in the data. NumPy and Pandas do not treat every hole the same way. `None` is Python's own missing value. `NaN` is a missing number. A text file can also mark a hole with a symbol such as `?`. That symbol stays as text until you tell Pandas to read it as missing.

    This lesson finds those holes and removes the rows or columns that contain them. Filling a hole with a substitute value is the next lesson.

    ## Datasets Used

    - **Small arrays.** One array contains `None`. Another contains `NaN`.
    - **Students.** Five rows, with missing age, sex, and GPA.
    - **Automobile Dataset** from the UCI Machine Learning Repository. Missing numbers are written as `?`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `None` in a NumPy array

    `None` is Python's missing value. NumPy stores an array that contains `None` with dtype `object`, because the values are not all numbers.
    """)
    return


@app.cell
def _(np):
    values = np.array([1, 2, None, 3, 4, 5])
    print(values)
    print(values.dtype)
    return (values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An object array cannot be converted to integers. The next cell raises an error. `None` is not a number, so it cannot become an `int`.
    """)
    return


@app.cell
def _(values):
    values.astype(int)  # This raises an error.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same array has no `mean`. The next cell raises an error. NumPy cannot add `None` to the numbers.
    """)
    return


@app.cell
def _(values):
    values.mean()  # This raises an error.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `NaN` in a NumPy array

    `NaN` means "not a number". It is a floating-point value, so NumPy stores the array as a float.
    """)
    return


@app.cell
def _(np):
    values2 = np.array([1, 2, np.nan, 3, 4, 5])
    print(values2)
    print(values2.dtype)
    return (values2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `min`, `max`, and `mean` run without an error. Each result is `NaN`, because one missing value makes the whole result missing.
    """)
    return


@app.cell
def _(values2):
    print(values2.min())
    print(values2.max())
    print(values2.mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NumPy has aggregations that skip `NaN`. `np.nanmin`, `np.nanmax`, and `np.nanmean` use the five numbers and ignore the hole. The mean is `(1 + 2 + 3 + 4 + 5) / 5 = 3`.
    """)
    return


@app.cell
def _(np, values2):
    print(np.nanmin(values2))
    print(np.nanmax(values2))
    print(np.nanmean(values2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The array below contains `None`. Change `None` to `np.nan`, then run the cell again. What is the dtype, and what does `np.nanmean` return?
    """)
    return


@app.cell
def _(np):
    with_none = np.array([10, 20, None, 40])
    print(with_none.dtype)
    return (with_none,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
with_none = np.array([10, 20, np.nan, 40])
print(with_none.dtype)
print(np.nanmean(with_none))
```
The dtype is `float64`. `np.nanmean` skips the hole and returns `23.333...`, which is `(10 + 20 + 40) / 3`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same values in a DataFrame

    Pandas keeps the `None` column as `object` and the `NaN` column as `float64`.
    """)
    return


@app.cell
def _(pd, values, values2):
    dfv = pd.DataFrame({"values": values, "values2": values2})
    print(dfv)
    return (dfv,)


@app.cell
def _(dfv):
    print(dfv.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pandas aggregations skip missing values. Both columns have the same five numbers, so both means are 3 and both sums are 15. The `object` column is included because its other entries are numbers.
    """)
    return


@app.cell
def _(dfv):
    print(dfv.mean())
    return


@app.cell
def _(dfv):
    print(dfv.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding missing values

    The students table has five rows. Age, sex, and GPA each have at least one hole. `studentID` is complete.
    """)
    return


@app.cell
def _(np, pd):
    students = [
        ["st_100", 17, "M", 3.7],
        ["st_101", 17, "M", np.nan],
        ["st_102", np.nan, "M", 2.4],
        ["st_103", np.nan, "F", np.nan],
        ["st_104", 19, np.nan, 3],
    ]
    df = pd.DataFrame(students, columns=["studentID", "Age", "Sex", "GPA"])
    print(df)
    return df, students


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `isnull` is `True` where a value is missing and `False` where a value is present.
    """)
    return


@app.cell
def _(df):
    print(df.isnull())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Summing the `True` values counts the holes. A column sum counts holes in that column. Age has 2, Sex has 1, and GPA has 2.
    """)
    return


@app.cell
def _(df):
    print(df.isnull().sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `axis=1` sums across the row. `st_100` has no holes. `st_103` has two.
    """)
    return


@app.cell
def _(df):
    print(df.isnull().sum(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `notnull` is the opposite of `isnull`. Its sums count the values that are present.
    """)
    return


@app.cell
def _(df):
    print(df.notnull())
    return


@app.cell
def _(df):
    print(df.notnull().sum())
    return


@app.cell
def _(df):
    print(df.notnull().sum(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Count the missing values in `Age`. Replace `None` with your code. How many rows have a missing age?
    """)
    return


@app.cell
def _():
    student_missing_age = None
    print(student_missing_age)
    return (student_missing_age,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_missing_age = df["Age"].isnull().sum()
```
**Two rows.** `st_102` and `st_103` have no age.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dropping rows and columns

    `dropna` removes every row that contains a missing value. `axis=1` removes every column that contains a missing value. Only `studentID` would remain, and that column is not enough for an analysis of the students.

    If one column is required and only a few rows are missing it, dropping those rows can be reasonable. Dropping a whole column throws away every value in that column, including the ones that are present.
    """)
    return


@app.cell
def _(df):
    print(df.dropna())
    return


@app.cell
def _(df):
    print(df.dropna(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `dropna` returns a new DataFrame. `df` still has every row and every missing value.
    """)
    return


@app.cell
def _(df):
    print(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Drop only the rows where `GPA` is missing. A missing age or a missing sex should stay. Replace `None` with your code. Which student IDs remain?
    """)
    return


@app.cell
def _():
    student_gpa_rows = None
    print(student_gpa_rows)
    return (student_gpa_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_gpa_rows = df.dropna(subset=["GPA"])
```
**`st_100`, `st_102`, and `st_104`.** `subset` limits the check to `GPA`. `st_102` stays, even though the age is missing.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `how` and `thresh`

    `how` has two choices. The default, `'any'`, drops a row or column when any value is missing. `'all'` drops it only when every value is missing.

    The new column `allNaN` is missing in every row. This table has 5 columns, so it is shown as HTML.
    """)
    return


@app.cell
def _(df, mo, np):
    df_all_nan = df.copy()
    df_all_nan["allNaN"] = np.nan
    mo.Html(df_all_nan.to_html())
    return (df_all_nan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `axis=1` and `how='all'` remove `allNaN` and keep the other columns. `df_all_nan` itself is unchanged.
    """)
    return


@app.cell
def _(df_all_nan):
    print(df_all_nan.dropna(axis=1, how="all"))
    return


@app.cell
def _(df_all_nan, mo):
    mo.Html(df_all_nan.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `thresh` is a minimum number of present values. `thresh=3` keeps a row that has at least 3 present values. `st_103` has only 2 (`studentID` and `Sex`), so that row is removed. The other four rows stay.
    """)
    return


@app.cell
def _(df_all_nan):
    print(df_all_nan.dropna(thresh=3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The cell keeps rows with at least 3 present values. Change `3` to `4` and run the cell again. Which student ID remains?
    """)
    return


@app.cell
def _(df_all_nan):
    kept_ids = df_all_nan.dropna(thresh=3)
    print(kept_ids["studentID"])
    return (kept_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
kept_ids = df_all_nan.dropna(thresh=4)
```
**`st_100`.** That row has four present values: the ID, the age, the sex, and the GPA. `allNaN` is missing, so it does not add to the count. Every other row has fewer than four present values.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same rule works on columns. `thresh=3` and `axis=1` keep a column with at least 3 present values. `Age`, `Sex`, and `GPA` each have at least 3. `allNaN` has 0, so it is the column that goes.
    """)
    return


@app.cell
def _(df_all_nan):
    print(df_all_nan.dropna(thresh=3, axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, `df_all_nan` is unchanged. `inplace=True` changes the DataFrame you call it on. The cell below copies the table first, then drops every column with no present values. The copy loses `allNaN`. The original `df_all_nan` still has that column.
    """)
    return


@app.cell
def _(df_all_nan):
    df_kept = df_all_nan.copy()
    df_kept.dropna(thresh=1, axis=1, inplace=True)
    print(df_kept)
    return (df_kept,)


@app.cell
def _(df_all_nan, mo):
    mo.Html(df_all_nan.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile Dataset

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a copy of the Automobile file stored on GitHub. GitHub allows a browser page to read its files, so the copy works in both places. The original UCI address is kept in a comment.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The file has no header row. `names` supplies the column names. `na_values='?'` turns the question marks into missing values while the file is read. `normalized_losses` is then `float64`, holes included.

    The table has 26 columns, so the first rows are shown as HTML.
    """)
    return


@app.cell
def _(auto_file, headers, mo, pd):
    # Reading from the UCI website (works when Python runs on your computer):
    # dfa = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    dfa = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(dfa.head().to_html())
    return (dfa,)


@app.cell
def _(dfa):
    print(dfa.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are the columns that still have holes, and the number of holes in each one. `normalized_losses` has 41.
    """)
    return


@app.cell
def _(dfa):
    missing_counts = dfa.isnull().sum()
    print(missing_counts[missing_counts > 0])
    return (missing_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Leave out `na_values='?'` and the question marks stay as text. `normalized_losses` is then `object`, because a column that mixes numbers and `?` is stored as text.
    """)
    return


@app.cell
def _(auto_file, headers, mo, pd):
    # dfb = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    # )
    dfb = pd.read_csv(auto_file, header=None, names=headers)
    mo.Html(dfb.head().to_html())
    return (dfb,)


@app.cell
def _(dfb):
    print(dfb.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `replace` can turn `?` into `NaN` after the file is read. It returns a new Series. That Series is still `object`: the remaining entries are text such as `"164"`, with `NaN` where `?` used to be. The column inside `dfb` changes only if you assign that Series back. The cells after the new Series show `dfb` and its dtypes, which still contain `?` and still say `object`.

    `na_values='?'` does both jobs while reading. The question marks become missing, and a column of numbers becomes `float64`.
    """)
    return


@app.cell
def _(dfb, np):
    replaced_losses = dfb.normalized_losses.replace(to_replace="?", value=np.nan)
    print(replaced_losses.head(10))
    print(replaced_losses.dtype)
    return (replaced_losses,)


@app.cell
def _(dfb, mo):
    mo.Html(dfb.head().to_html())
    return


@app.cell
def _(dfb):
    print(dfb.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Copy `dfb` and assign the replaced `normalized_losses` column back to the copy. Replace `None` with your code. How many missing values does that column have?
    """)
    return


@app.cell
def _():
    student_fixed_losses = None
    print(student_fixed_losses)
    return (student_fixed_losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_fixed_losses = dfb.copy()
student_fixed_losses["normalized_losses"] = student_fixed_losses[
    "normalized_losses"
].replace("?", np.nan)
print(student_fixed_losses["normalized_losses"].isnull().sum())
```
**41.** That matches the count from the file that was read with `na_values='?'`. The column is still `object`, because the numbers are text. `na_values='?'` at read time marks `?` as missing in every column and lets the numeric columns stay numeric.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - An array that contains `None` has dtype `object`. `astype(int)` and `mean` raise an error.
    - `NaN` is a floating-point missing value. `min`, `max`, and `mean` return `NaN`. `np.nanmean` skips the missing values.
    - In Pandas, `mean` and `sum` skip missing values.
    - `isnull` and `notnull` locate the holes. `dropna` removes rows or columns and returns a new DataFrame. The original stays the same unless you assign the result or use `inplace=True`.
    - `how='all'` removes a row or column only when every value is missing. `thresh` is the smallest number of present values that lets a row or column stay.
    - In the Automobile file, `?` is text. `na_values='?'` reads it as missing, and a numeric column stays numeric. `replace` returns a new Series that is still text, with `NaN` in place of `?`. The column changes only when that Series is assigned back.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What dtype does NumPy choose for an array that contains `None`?
    2. What does `mean` return for an array that contains `NaN`?
    3. What does `np.nanmean` do with a `NaN`?
    4. Do `mean` and `sum` in Pandas use the missing values?
    5. What does `dropna()` remove when `how` is left at its default?
    6. What does `thresh=3` require?
    7. Why is `normalized_losses` an object when the file is read without `na_values='?'`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **`object`:** The array mixes numbers and `None`, so NumPy does not store it as a numeric dtype.
2. **`NaN`:** One missing value makes `min`, `max`, and `mean` missing too.
3. **It skips it:** `np.nanmean` averages the numbers that are present.
4. **No:** Pandas aggregations skip missing values.
5. **Any row with a hole:** The default is `how='any'`, so one missing value is enough to remove the row.
6. **At least 3 present values:** A row or column with fewer present values is removed.
7. **`?` is text:** A column that mixes numbers and question marks is stored as text. `na_values='?'` marks those question marks as missing while the file is read, and the numeric column stays numeric.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - VanderPlas, J. (2017). *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media. Chapter 3.
    """)
    return


if __name__ == "__main__":
    app.run()
