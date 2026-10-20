# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
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
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fixing formatting and removing duplicates

    ## Objectives

    - Strip extra spaces from names and give each name one capitalization.
    - Replace a value that cannot be real, such as an age of 125 or a score of -1, with a missing value.
    - Turn the sex labels into `m` and `f`, and mark `_` as missing.
    - Remove duplicate rows with `drop_duplicates`, and choose which copy to keep.

    ## Background

    The same name can arrive as ` John`, `john`, and `John `. A column of sexes can mix `MALE`, ` Fem`, and `_`. An age of 125 in a table of people in their twenties is not usable until someone checks the source. Duplicate rows make a person count as two people.

    This lesson cleans one small table by hand: text first, then impossible numbers, then duplicate rows.

    ## Datasets Used

    - **A made-up table** of 13 people, with messy names, sexes, one impossible age, one impossible score, and repeated rows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    Spaces sit on one side of some names, or on both sides. Capitalization disagrees. Sex is written as a full word, a fragment, or `_`. One age is 125. One score is -1.
    """)
    return


@app.cell
def _(pd):
    people = {
        "Name": [
            " John",
            "annA",
            "mary",
            " mary ",
            " AnnA",
            " charles",
            "John ",
            " RoB",
            "Rob ",
            " john",
            "anna ",
            " Charles",
            " Heather ",
        ],
        "Age": [26, 23, 23, 23, 23, 24, 26, 24, 22, 26, 23, 24, 125],
        "Sex": [
            "MALE",
            " female",
            "_",
            "F",
            " Fem",
            "M",
            " male ",
            " M  ",
            " M",
            " Mal",
            " Fem",
            " male ",
            " f ",
        ],
        "Score": [85, 31, 55, 55, 31, 77, 85, 63, 63, 85, -1, 77, 80],
    }
    df = pd.DataFrame(people)
    print(df)
    return df, people


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Names

    `str.strip` removes spaces from both ends of each name. It does not change spaces in the middle, and these names have none. `str.contains(' ')` is `True` when a space remains.
    """)
    return


@app.cell
def _(df):
    print(df["Name"].str.contains(" "))
    return


@app.cell
def _(df):
    stripped_names = df["Name"].str.strip()
    print(stripped_names)
    return (stripped_names,)


@app.cell
def _(stripped_names):
    print(stripped_names.str.contains(" "))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `str.capitalize` makes the first letter uppercase and the rest lowercase. `annA` becomes `Anna`. `RoB` becomes `Rob`.
    """)
    return


@app.cell
def _(stripped_names):
    clean_names = stripped_names.str.capitalize()
    print(clean_names)
    return (clean_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The names are written back into a copy. `df` stays messy, so the later steps can show what each change did.
    """)
    return


@app.cell
def _(clean_names, df):
    df_named = df.copy()
    df_named["Name"] = clean_names
    print(df_named)
    return (df_named,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Strip and capitalize `'  aNnA '`. Replace `None` with your code. What is the result?
    """)
    return


@app.cell
def _():
    student_name = None
    print(student_name)
    return (student_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
import pandas as pd

student_name = pd.Series(["  aNnA "]).str.strip().str.capitalize()
```
**`Anna`.** `strip` removes the spaces. `capitalize` keeps `A` and lowers `NnA`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Age

    The ages sit in the twenties, except for 125. The boxplot draws that value far from the box.
    """)
    return


@app.cell
def _(df_named):
    print(df_named.Age.describe())
    return


@app.cell
def _(df_named, plt, sns):
    _figure, _axes = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df_named.Age, ax=_axes)
    _axes.set_title("Age, including 125")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    125 is Heather's age. It is not a plausible age next to the other rows, and this lesson has no source to check. The cell replaces 125 with a missing value. A mean would hide the hole. The hole is the reminder that the source still needs to be checked.
    """)
    return


@app.cell
def _(df_named, np):
    df_aged = df_named.copy()
    df_aged["Age"] = df_aged.Age.replace(125, np.nan)
    print(df_aged)
    return (df_aged,)


@app.cell
def _(df_aged, plt, sns):
    _figure, _axes = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df_aged.Age, ax=_axes)
    _axes.set_title("Age, with 125 removed")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sex

    The bar chart shows a different label for almost every row. `MALE`, ` male `, `M`, and ` Mal` are the same category written in different ways.
    """)
    return


@app.cell
def _(df_aged, plt):
    _figure, _axes = plt.subplots(figsize=(8, 3))
    df_aged.Sex.value_counts().plot(kind="bar", rot=45, ax=_axes)
    _axes.set_title("Sex labels before cleaning")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `strip` removes the extra spaces. `lower` makes every letter lowercase. The first character is then `m`, `f`, or `_`.
    """)
    return


@app.cell
def _(df_aged):
    sex = df_aged.Sex.str.strip()
    print(sex)
    return (sex,)


@app.cell
def _(sex):
    sex_lower = sex.str.lower()
    print(sex_lower)
    return (sex_lower,)


@app.cell
def _(pd, sex_lower):
    sex_letter = pd.Series([text[0] for text in sex_lower])
    print(sex_letter)
    return (sex_letter,)


@app.cell
def _(plt, sex_letter):
    _figure, _axes = plt.subplots(figsize=(4, 3))
    sex_letter.value_counts().plot(kind="bar", rot=0, ax=_axes)
    _axes.set_title("First letter, including _")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `_` is not a sex. It is the third row, index 2, which is Mary. That entry becomes a missing value. The new Series uses the positions 0, 1, 2, ..., so assigning at index 2 changes that row.
    """)
    return


@app.cell
def _(np, sex_letter):
    sex_missing = sex_letter.copy()
    sex_missing[2] = np.nan
    print(sex_missing)
    return (sex_missing,)


@app.cell
def _(df_aged, sex_missing):
    df_sexed = df_aged.copy()
    df_sexed["Sex"] = sex_missing
    print(df_sexed)
    return (df_sexed,)


@app.cell
def _(df_sexed, plt):
    _figure, _axes = plt.subplots(figsize=(4, 3))
    df_sexed.Sex.value_counts().plot(kind="bar", rot=0, ax=_axes)
    _axes.set_title("Sex after cleaning")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The missing sex could be filled with the most common letter, or guessed from the name Mary. The better fix is the source. The hole stays.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Mark every `_` as missing with `replace`, instead of assigning index 2. Replace `None` with your code. Which row changes?
    """)
    return


@app.cell
def _():
    student_sex = None
    print(student_sex)
    return (student_sex,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_sex = sex_letter.replace("_", np.nan)
```
**Index 2, Mary.** `replace` finds the underscore wherever it sits. Assigning index 2 works here only because that is the row that contains `_`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Score

    The score -1 sits away from the other scores. It belongs to the third Anna, at index 10. A score on this table is not negative, and there is no source to check, so -1 becomes missing.
    """)
    return


@app.cell
def _(df_sexed, plt, sns):
    _figure, _axes = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df_sexed.Score, ax=_axes)
    _axes.set_title("Score, including -1")
    return _figure


@app.cell
def _(df_sexed, np):
    df_scored = df_sexed.copy()
    df_scored["Score"] = df_scored.Score.replace(-1, np.nan)
    print(df_scored)
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Duplicate rows

    `drop_duplicates` compares the whole row. The default is `keep='first'`, which keeps the first copy and drops the later ones. These rows are repeated:

    - John, 26, `m`, 85 appears three times.
    - Anna, 23, `f`, 31 appears twice.
    - Charles, 24, `m`, 77 appears twice.

    The two Mary rows are not duplicates: one sex is missing and the other is `f`. The two Rob rows are not duplicates: the ages are 24 and 22. Anna at index 10 has a missing score, so she is not a copy of the other Annas.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Writing `keep='first'` is the same call.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates(keep="first"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `df_scored` is unchanged.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `keep='last'` keeps the last copy. John stays at index 9, Anna at index 4, and Charles at index 11.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates(keep="last"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `keep=False` drops every row that has a copy. Both Mary rows stay, both Rob rows stay, Anna with the missing score stays, and Heather stays. John, the Annas with score 31, and Charles all go.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates(keep=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A list of column names limits the comparison. `['Name']` keeps one row per name. With the default `keep='first'`, Mary stays as the row with the missing sex, Rob stays at age 24, and Anna stays with score 31.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates(["Name"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `keep='last'` keeps the other copy. Rob stays at age 22. Anna stays as the row with the missing score, and the Annas with score 31 are dropped. Mary stays as the row whose sex is `f`. The two Rob rows may be the same person or two people. The name alone cannot decide that.
    """)
    return


@app.cell
def _(df_scored):
    print(df_scored.drop_duplicates(["Name"], keep="last"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Drop duplicate names and keep the last copy. Replace `None` with your code. What age does Rob have in the result?
    """)
    return


@app.cell
def _():
    student_last_name = None
    print(student_last_name)
    return (student_last_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_last_name = df_scored.drop_duplicates(["Name"], keep="last")
```
**22.** That is index 8. The Rob at index 7 is 24, and `keep='last'` drops him.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `inplace=True` changes the DataFrame you call it on. The copy below keeps the first row of each name. `df_scored` is still the full cleaned table.
    """)
    return


@app.cell
def _(df_scored):
    df_by_name = df_scored.copy()
    df_by_name.drop_duplicates(["Name"], inplace=True)
    print(df_by_name)
    return (df_by_name,)


@app.cell
def _(df_scored):
    print(len(df_scored))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ignore_index=True` numbers the remaining rows from 0. The names are already unique, so no further row is dropped. The index is what changes.
    """)
    return


@app.cell
def _(df_by_name):
    df_reset = df_by_name.copy()
    df_reset.drop_duplicates(ignore_index=True, inplace=True)
    print(df_reset)
    return (df_reset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - `str.strip` removes spaces at the ends of a name. `str.capitalize` makes the first letter uppercase and the rest lowercase, so `annA` and ` Anna ` both become `Anna`.
    - An age of 125 and a score of -1 are replaced with missing values. The other rows give no support for those numbers, and a filled-in mean would hide the problem. The source is the place to recover them.
    - Sex is reduced to `m` or `f`. `_` is missing. Guessing it from the name Mary is possible, and checking the source is better.
    - `drop_duplicates` compares the columns you name, or the whole row when you name none. `keep='first'` and `keep='last'` choose the copy that stays. `keep=False` drops every copy.
    - Two rows can share a name and still both remain when the whole row is compared. The Rob rows have different ages. The Mary rows have different sexes.
    - `drop_duplicates` returns a new DataFrame unless `inplace=True`. `ignore_index=True` restarts the index at 0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does `str.strip` remove from a name?
    2. What does `str.capitalize` do to the letters after the first?
    3. Why is Heather's age replaced with a missing value?
    4. What do `m` and `f` stand for after the sex column is cleaned?
    5. Which row does `drop_duplicates()` keep when a row is repeated?
    6. What does `keep=False` do with a repeated row?
    7. Why do both Rob rows survive `drop_duplicates()` when no columns are named?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Spaces at both ends:** Spaces in the middle would stay. These names have none.
2. **It lowercases them:** `annA` becomes `Anna`, and `RoB` becomes `Rob`.
3. **125 is not plausible here:** The other ages are in the twenties, and the lesson has no source to check. Leaving a hole keeps that problem visible.
4. **Male and female:** Each label was reduced to its first letter. `_` was marked missing.
5. **The first copy:** That is `keep='first'`, the default.
6. **It drops every copy:** A row stays only when it appears once.
7. **The ages differ:** One Rob is 24 and the other is 22, so the full rows are not duplicates.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Camm, J. D., Cochran, J. J., Fry, M. J., Ohlmann, J. W., Anderson, D. R., Sweeney, D. J., and Williams, T. A. (2015). *Essentials of Business Analytics*. Cengage Learning.
    """)
    return


if __name__ == "__main__":
    app.run()
