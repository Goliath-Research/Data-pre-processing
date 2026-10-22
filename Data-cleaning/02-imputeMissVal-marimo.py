# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
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
    import numpy as np
    import pandas as pd

    pd.set_eng_float_format(accuracy=2, use_eng_prefix=True)

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imputing Missing Values

    ## Objectives

    - Replace missing values with a constant using `fillna`.
    - Copy a neighboring value with `ffill` and `bfill`, down a column or across a row.
    - Fill a numeric column with its mean or its median, and a text column with its most common value.
    - Estimate a missing number from the other numbers in the same column with `interpolate`.
    - Estimate a missing number from similar rows with `KNNImputer`.

    ## Background

    Imputation replaces a missing value with a substitute so the row can stay in the table. The substitute is a guess. A constant such as 0 is quick to write and can be far from the other values. A mean, a median, or a neighbor uses the data that is already there. None of these guesses recovers the value that was never recorded. When the real value can be found, that is the better fix.

    ## Datasets Used

    - **Students.** Five rows, with missing age, sex, and GPA. Used for constants, neighbors, the mean, the median, and the mode.
    - **Daily production.** Ten dates, with holes in `ProdA` and `ProdB`. Used for interpolation and for k-nearest neighbors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A constant

    `fillna` returns a new DataFrame. The students table below is the one every later section starts from.
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
    Filling every hole with 0 keeps the shape of the table. An age of 0 and a GPA of 0 were not in the original data.
    """)
    return


@app.cell
def _(df):
    print(df.fillna(0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `df` is unchanged.
    """)
    return


@app.cell
def _(df):
    print(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Any constant works the same way. Here every hole becomes 100. That number is not an age, a sex, or a GPA in this table.
    """)
    return


@app.cell
def _(df):
    print(df.fillna(100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Fill the missing ages with 18 and leave `Sex` and `GPA` as they are. Replace `None` with your code. Which rows change?
    """)
    return


@app.cell
def _():
    student_age_18 = None
    print(student_age_18)
    return (student_age_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_age_18 = df.copy()
student_age_18["Age"] = student_age_18["Age"].fillna(18)
```
**`st_102` and `st_103`.** Those are the two rows with a missing age. The missing sex and the missing GPAs stay missing.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward fill and back fill

    `ffill` copies the last present value down the column. A missing value in the first row stays missing, because there is no earlier value to copy. The order of the rows matters: sorting the table first would copy different neighbors.
    """)
    return


@app.cell
def _(df):
    print(df.ffill())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The missing age in `st_102` becomes 17, copied from `st_101`. The missing GPA in `st_103` becomes 2.4, copied from `st_102`. The missing sex in `st_104` becomes `F`, copied from `st_103`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `axis=1` copies across the row, from left to right. A neighbor can then be a different kind of value. `st_102` gets the student ID in the age column. `st_104` gets the age in the sex column. `st_101` gets `M` in the GPA column.
    """)
    return


@app.cell
def _(df):
    print(df.ffill(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bfill` copies the next present value upward. The missing sex in the last row stays missing, because there is no later sex to copy. The missing age in `st_102` becomes 19, copied from `st_104`.
    """)
    return


@app.cell
def _(df):
    print(df.bfill())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Across the row, `bfill` copies from right to left. `st_102` gets `M` in the age column, and `st_104` gets the GPA in the sex column.
    """)
    return


@app.cell
def _(df):
    print(df.bfill(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Look at `df.ffill()` again. The sex of `st_104` was missing. Which value was copied into it, and from which row?
    """)
    return


@app.cell
def _(df):
    sex_filled = df.ffill()
    print(sex_filled["Sex"])
    return (sex_filled,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
**`F`, from `st_103`.** Forward fill copies the last present value in the column. `st_103` is the row just above `st_104`, and its sex is `F`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mean, the median, and the mode

    A measure of center is a more careful constant. It is still one number, repeated for every hole in that column. The mean and the median are for numbers. The students table uses them on `Age` and `GPA` only.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The mean age of the three present ages is about 17.67. The mean GPA of the three present GPAs is about 3.03. Missing values are skipped.
    """)
    return


@app.cell
def _(df):
    print(df[["Age", "GPA"]].mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `fillna` can take a Series. Each column is filled with its own mean. The other columns are then copied back so the table has an ID and a sex again.
    """)
    return


@app.cell
def _(df):
    df_mean = df.copy()
    df_mean[["Age", "GPA"]] = df[["Age", "GPA"]].fillna(df[["Age", "GPA"]].mean())
    print(df_mean)
    return (df_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The median age is 17, and the median GPA is 3. The median is the middle present value, so one unusually high or low value does not pull it the way it pulls the mean.
    """)
    return


@app.cell
def _(df):
    print(df[["Age", "GPA"]].median())
    return


@app.cell
def _(df):
    df_median = df.copy()
    df_median[["Age", "GPA"]] = df[["Age", "GPA"]].fillna(df[["Age", "GPA"]].median())
    print(df_median)
    return (df_median,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Sex` is text, so the mean does not apply. `value_counts` shows that `M` appears three times and `F` appears once. The most common value is `M`.

    Filling a missing sex with `M` is only a demonstration. A missing sex should be looked up in the source.
    """)
    return


@app.cell
def _(df):
    print(df.Sex.value_counts())
    return


@app.cell
def _(df):
    print(df.Sex.fillna("M"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Fill the missing sex with the value `mode` returns. Replace `None` with your code. Which student ID changes?
    """)
    return


@app.cell
def _():
    student_sex_mode = None
    print(student_sex_mode)
    return (student_sex_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_sex_mode = df.copy()
student_sex_mode["Sex"] = student_sex_mode["Sex"].fillna(
    student_sex_mode["Sex"].mode()[0]
)
```
**`st_104`.** `mode` returns `M`, the most common sex. That is the same fill as writing `"M"`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A value you calculate

    `fillna` can also take one number that you calculated yourself. This function returns the halfway point between the smallest present GPA and the largest. That halfway point is `(2.4 + 3.7) / 2 = 3.05`.
    """)
    return


@app.cell
def _(pd):
    def mean_ext(column):
        present = pd.Series(column)
        return (present.min() + present.max()) / 2

    return (mean_ext,)


@app.cell
def _(df, mean_ext):
    print(mean_ext(df.GPA))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `fillna` and `replace` both put 3.05 where `GPA` is missing. `replace` is aimed at particular values. Here the value to find is `NaN`. Neither call changes `df`.
    """)
    return


@app.cell
def _(df, mean_ext):
    print(df.GPA.fillna(mean_ext(df.GPA)))
    return


@app.cell
def _(df, mean_ext, np):
    print(df.GPA.replace(np.nan, mean_ext(df.GPA)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpolation

    `interpolate` estimates a hole from the numbers around it in the same column. That fits a series that is already in order, such as production recorded by date. The default method is linear: a hole halfway between two numbers becomes their average.

    In `ProdA`, the hole at index 2 sits between 21 and 21, so it becomes 21. The hole at index 5 sits between 12 and 12, so it becomes 12. The hole at index 8 sits between 20 and 30, so it becomes 25.
    """)
    return


@app.cell
def _(np, pd):
    dates = pd.date_range(start="2022-05-05", periods=10).to_pydatetime().tolist()
    prod_a = [16, 21, np.nan, 21, 12, np.nan, 12, 20, np.nan, 30]
    prod_b = [36, 38, np.nan, 42, np.nan, 60, 47, 67, 73, 55]
    dfd = pd.DataFrame({"Date": dates, "ProdA": prod_a, "ProdB": prod_b})
    print(dfd)
    return dates, dfd, prod_a, prod_b


@app.cell
def _(dfd):
    print(dfd.isnull().sum())
    return


@app.cell
def _(dfd):
    print(dfd.ProdA.interpolate())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Naming `method='linear'` is the same call. The default is already linear.
    """)
    return


@app.cell
def _(dfd):
    print(dfd.ProdA.interpolate(method="linear"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `method='polynomial'` with `order=2` fits a curve instead of a straight line between neighbors. The hole at index 2 is then about 23.67, the hole at index 5 is about 8.66, and the hole at index 8 is about 26.35. A different order of the rows would produce a different curve. A time series is kept in date order, so the row order is part of the data.
    """)
    return


@app.cell
def _(dfd):
    print(dfd.ProdA.interpolate(method="polynomial", order=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Fill the holes in `ProdB` with linear interpolation. Replace `None` with your code. What value appears at index 2, between 38 and 42?
    """)
    return


@app.cell
def _():
    student_prod_b = None
    print(student_prod_b)
    return (student_prod_b,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_prod_b = dfd.ProdB.interpolate()
```
**40.** Index 2 is halfway from 38 to 42. Index 4, between 42 and 60, becomes 51.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k nearest neighbors

    `KNNImputer` looks at the other rows. For `n_neighbors=1`, a row with one present value borrows the other column from the closest row. Distance here uses the columns you pass in, `ProdA` and `ProdB`.

    - Index 4 has `ProdA` of 12 and a missing `ProdB`. The closest `ProdA` is the other 12, at index 6, where `ProdB` is 47. The estimate is 47.
    - Index 5 has `ProdB` of 60 and a missing `ProdA`. The closest `ProdB` is 55, at index 9, where `ProdA` is 30. The estimate is 30.
    - Index 2 is missing both values. There is no value to compare, so each estimate is the mean of the present values in that column: about 18.86 for `ProdA` and 52.25 for `ProdB`.
    """)
    return


@app.cell
def _():
    from sklearn.impute import KNNImputer

    return (KNNImputer,)


@app.cell
def _(KNNImputer, dfd):
    imputer_1 = KNNImputer(n_neighbors=1)
    imputed_1 = imputer_1.fit_transform(dfd[["ProdA", "ProdB"]])
    dfd_1 = dfd.copy()
    dfd_1["ProdA_1"] = imputed_1[:, 0]
    dfd_1["ProdB_1"] = imputed_1[:, 1]
    print(dfd_1[["ProdA", "ProdB", "ProdA_1", "ProdB_1"]])
    return dfd_1, imputed_1, imputer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `n_neighbors=2`, each estimate is the mean of the two closest rows. Index 4 gets a `ProdB` of 41.5, and index 5 gets a `ProdA` of 25. Index 2 does not change: both values are missing, so the column means are used again. This table has 6 columns, so it is shown as HTML.
    """)
    return


@app.cell
def _(KNNImputer, dfd_1, mo):
    imputer_2 = KNNImputer(n_neighbors=2)
    imputed_2 = imputer_2.fit_transform(dfd_1[["ProdA", "ProdB"]])
    dfd_2 = dfd_1.copy()
    dfd_2["ProdA_2"] = imputed_2[:, 0]
    dfd_2["ProdB_2"] = imputed_2[:, 1]
    mo.Html(
        dfd_2[
            ["ProdA", "ProdB", "ProdA_1", "ProdB_1", "ProdA_2", "ProdB_2"]
        ].to_html()
    )
    return dfd_2, imputed_2, imputer_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `n_neighbors=3` averages three neighbors. Index 4 gets a `ProdB` of 50, and index 5 gets a `ProdA` of about 20.67. Index 2 is still the two column means. More neighbors pull the estimate toward the rest of the column.
    """)
    return


@app.cell
def _(KNNImputer, dfd_2, mo):
    imputer_3 = KNNImputer(n_neighbors=3)
    imputed_3 = imputer_3.fit_transform(dfd_2[["ProdA", "ProdB"]])
    dfd_3 = dfd_2.copy()
    dfd_3["ProdA_3"] = imputed_3[:, 0]
    dfd_3["ProdB_3"] = imputed_3[:, 1]
    mo.Html(dfd_3.drop(columns="Date").to_html())
    return dfd_3, imputed_3, imputer_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - `fillna` returns a new DataFrame. A constant such as 0 or 100 fills every hole, including holes in a text column.
    - `ffill` copies the previous present value. `bfill` copies the next one. Down a column, this follows the row order. Across a row, the neighbor can be a different kind of value.
    - The mean and the median fill a numeric column with one measure of center. The most common value can fill a text column. Each of these is one repeated guess.
    - `interpolate` estimates a hole from the other numbers in the same column. Linear interpolation averages the neighbors. A polynomial fit uses a curve. The row order changes the result.
    - `KNNImputer` borrows from the closest rows. A row that is missing every column used for the distance gets the mean of each column.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Does `df.fillna(0)` change `df`?
    2. What does `ffill` do with a missing value in the first row of a column?
    3. Why is `ffill(axis=1)` a poor way to fill the students table?
    4. In this table, how does the filled age from the mean differ from the filled age from the median?
    5. What does linear interpolation put between two equal neighbors?
    6. With `n_neighbors=1`, where does the `ProdB` estimate for index 4 come from?
    7. What does `KNNImputer` use when `ProdA` and `ProdB` are both missing?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **No:** `fillna` returns a new DataFrame. `df` still has the same holes.
2. **It leaves the hole:** There is no earlier value to copy.
3. **The columns are different kinds of values:** The fill can copy a student ID into `Age`, or a sex into `GPA`.
4. **17.67 and 17:** The mean of the present ages is about 17.67. The median is 17, the middle present age.
5. **That same number:** A hole between 21 and 21 becomes 21. A hole between 12 and 12 becomes 12.
6. **Index 6:** That row also has `ProdA` equal to 12, and its `ProdB` is 47.
7. **The column mean:** There is no present value in that row to compare with the other rows. `ProdA` is about 18.86 and `ProdB` is 52.25.
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
