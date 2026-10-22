# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
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
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Handling Outliers

    ## Objectives

    - Mark unusual values with a boxplot, and see how the whisker length `whis` changes that cutoff.
    - Compare a boxenplot, which draws more boxes than a boxplot.
    - Flag values more than 3 standard deviations from the mean with a z-score.
    - Look for an unusual pair of values on a scatter plot, and see which rows a horsepower cutoff removes.

    ## Background

    An outlier is a value far from the rest of the data. A boxplot marks points outside its whiskers. A longer whisker marks fewer points. A z-score measures distance from the mean in standard deviations. A scatter plot shows two columns at once, so a point can look ordinary in each column and unusual as a pair.

    A rule can flag a point. It cannot say whether the point is a mistake. That decision needs the source of the data.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. The charts use `horsepower`, `city_mpg`, and `highway_mpg`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile Dataset

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download the file from the UCI website.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks the download.

    This lesson reads a copy of the file stored on GitHub. GitHub allows a browser page to read its files, so the copy works in both places. The original UCI address is kept in a comment.

    The table has 26 columns, so the first rows are shown as HTML.
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
    # Reading from the UCI website (works when Python runs on your computer):
    # df = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    df = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(df.head().to_html())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rest of the lesson uses three numeric columns. Two horsepower values are missing.
    """)
    return


@app.cell
def _(df):
    df2 = df[["horsepower", "city_mpg", "highway_mpg"]]
    print(df2.head())
    return (df2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boxplots and the interquartile range

    A boxplot draws the median, the middle half of the data, and whiskers. Points beyond the whiskers are drawn one by one. The first chart shows all three columns. The next two horsepower charts are the same chart: the default whisker and `whis=1.5`.
    """)
    return


@app.cell
def _(df2, plt):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    df2.boxplot(ax=_axes)
    _axes.set_title("horsepower, city_mpg, highway_mpg")
    return _figure


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=df2.horsepower, ax=_axes)
    _axes.set_title("Horsepower")
    return _figure


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=df2.horsepower, whis=1.5, ax=_axes)
    _axes.set_title("Horsepower, whis=1.5")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `whis` is the number of interquartile ranges (IQRs) past the first and third quartiles. The IQR is the third quartile minus the first. A point beyond `quartile ± whis × IQR` is marked as an outlier.

    A smaller `whis` makes the whiskers shorter, so more points are marked. `whis=1` marks more horsepower values than `whis=1.5`.
    """)
    return


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=df2.horsepower, whis=1, ax=_axes)
    _axes.set_title("Horsepower, whis=1")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A larger `whis` makes the whiskers longer. `whis=2` marks only the two highest horsepower values.
    """)
    return


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=df2.horsepower, whis=2, ax=_axes)
    _axes.set_title("Horsepower, whis=2")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `whis=0` draws no whisker. Every point outside the box, the middle half of the data, is marked.
    """)
    return


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=df2.horsepower, whis=0, ax=_axes)
    _axes.set_title("Horsepower, whis=0")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The function below draws the boxplot and returns the values outside the whiskers. Quartiles skip missing horsepower, and a missing value fails both comparisons, so it is not listed as an outlier.
    """)
    return


@app.cell
def _(sns):
    def boxplot_outliers(data, whis=1.5, ax=None):
        axes = sns.boxplot(y=data, whis=whis, ax=ax)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data < q1 - whis * iqr) | (data > q3 + whis * iqr)]
        return axes, outliers

    return (boxplot_outliers,)


@app.cell
def _(boxplot_outliers, df2, plt):
    _figure, _axes = plt.subplots(figsize=(4, 5))
    _drawn, horsepower_outliers = boxplot_outliers(df2.horsepower, ax=_axes)
    _axes.set_title("Horsepower, whis=1.5")
    return _figure, horsepower_outliers


@app.cell
def _(horsepower_outliers):
    print(horsepower_outliers.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same three whisker lengths, side by side. `whis=1` marks 14 horsepower values. `whis=1.5` marks 6: 262, 200, 207, 207, 207, and 288. `whis=2` marks 262 and 288.
    """)
    return


@app.cell
def _(boxplot_outliers, df2, plt):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    _a, out_h1 = boxplot_outliers(df2.horsepower, whis=1, ax=_axes[0])
    _axes[0].set_title("Horsepower\nwhis=1")
    _b, out_h2 = boxplot_outliers(df2.horsepower, whis=1.5, ax=_axes[1])
    _axes[1].set_title("Horsepower\nwhis=1.5")
    _c, out_h3 = boxplot_outliers(df2.horsepower, whis=2, ax=_axes[2])
    _axes[2].set_title("Horsepower\nwhis=2")
    return _figure, out_h1, out_h2, out_h3


@app.cell
def _(out_h1, out_h2, out_h3):
    print("whis=1  ", out_h1.tolist())
    print("whis=1.5", out_h2.tolist())
    print("whis=2  ", out_h3.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The cell uses `whis=1.5` for horsepower. Change it to `2` and run the cell again. Which values are still marked?
    """)
    return


@app.cell
def _(boxplot_outliers, df2):
    _drawn, horsepower_whis = boxplot_outliers(df2.horsepower, whis=1.5)
    print(horsepower_whis.tolist())
    return (horsepower_whis,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
_drawn, horsepower_whis = boxplot_outliers(df2.horsepower, whis=2)
```
**262 and 288.** A longer whisker reaches 200 and 207, so those points are no longer marked.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### City fuel economy

    `city_mpg` has no missing values. `whis=1` marks 47, 49, and 45. `whis=1.5` marks 47 and 49. `whis=2` marks none.
    """)
    return


@app.cell
def _(boxplot_outliers, df2, plt):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    _a, out_c1 = boxplot_outliers(df2.city_mpg, whis=1, ax=_axes[0])
    _axes[0].set_title("City (mpg)\nwhis=1")
    _b, out_c2 = boxplot_outliers(df2.city_mpg, whis=1.5, ax=_axes[1])
    _axes[1].set_title("City (mpg)\nwhis=1.5")
    _c, out_c3 = boxplot_outliers(df2.city_mpg, whis=2, ax=_axes[2])
    _axes[2].set_title("City (mpg)\nwhis=2")
    return _figure, out_c1, out_c2, out_c3


@app.cell
def _(out_c1, out_c2, out_c3):
    print("whis=1  ", out_c1.tolist())
    print("whis=1.5", out_c2.tolist())
    print("whis=2  ", out_c3.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Highway fuel economy

    `whis=1` marks seven highway values, from 46 to 54. `whis=1.5` marks 53, 54, and 50. `whis=2` marks 53 and 54.
    """)
    return


@app.cell
def _(boxplot_outliers, df2, plt):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    _a, out_w1 = boxplot_outliers(df2.highway_mpg, whis=1, ax=_axes[0])
    _axes[0].set_title("Highway (mpg)\nwhis=1")
    _b, out_w2 = boxplot_outliers(df2.highway_mpg, whis=1.5, ax=_axes[1])
    _axes[1].set_title("Highway (mpg)\nwhis=1.5")
    _c, out_w3 = boxplot_outliers(df2.highway_mpg, whis=2, ax=_axes[2])
    _axes[2].set_title("Highway (mpg)\nwhis=2")
    return _figure, out_w1, out_w2, out_w3


@app.cell
def _(out_w1, out_w2, out_w3):
    print("whis=1  ", out_w1.tolist())
    print("whis=1.5", out_w2.tolist())
    print("whis=2  ", out_w3.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Use `boxplot_outliers` on `highway_mpg` with `whis=2`. Replace `None` with your code. Which values are marked?
    """)
    return


@app.cell
def _():
    student_highway = None
    print(student_highway)
    return (student_highway,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
_drawn, student_highway = boxplot_outliers(df2.highway_mpg, whis=2)
print(student_highway.tolist())
```
**53 and 54.** Those two cars have the highest highway fuel economy in the table.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boxenplots

    A boxenplot draws several boxes, one for each stretch of the tails, instead of one box and two whiskers. `k_depth` chooses how many boxes to draw.

    - `proportion` uses `outlier_prop`. The default proportion is 0.007, about 7 in 1,000.
    - `trustworthy` chooses a depth that the sample size can support.
    - `tukey` follows the letter-value rule from Tukey's boxplot.

    Points beyond the last box are drawn one by one. The three panels are the same column with those three choices.
    """)
    return


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    sns.boxenplot(
        y=df2.horsepower,
        k_depth="proportion",
        outlier_prop=0.007,
        ax=_axes[0],
    )
    _axes[0].set_title("Horsepower\nproportion, 0.007")
    sns.boxenplot(y=df2.horsepower, k_depth="trustworthy", ax=_axes[1])
    _axes[1].set_title("Horsepower\ntrustworthy")
    sns.boxenplot(y=df2.horsepower, k_depth="tukey", ax=_axes[2])
    _axes[2].set_title("Horsepower\ntukey")
    return _figure


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    sns.boxenplot(
        y=df2.city_mpg, k_depth="proportion", outlier_prop=0.007, ax=_axes[0]
    )
    _axes[0].set_title("City (mpg)\nproportion, 0.007")
    sns.boxenplot(y=df2.city_mpg, k_depth="trustworthy", ax=_axes[1])
    _axes[1].set_title("City (mpg)\ntrustworthy")
    sns.boxenplot(y=df2.city_mpg, k_depth="tukey", ax=_axes[2])
    _axes[2].set_title("City (mpg)\ntukey")
    return _figure


@app.cell
def _(df2, plt, sns):
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 5))
    sns.boxenplot(
        y=df2.highway_mpg, k_depth="proportion", outlier_prop=0.007, ax=_axes[0]
    )
    _axes[0].set_title("Highway (mpg)\nproportion, 0.007")
    sns.boxenplot(y=df2.highway_mpg, k_depth="trustworthy", ax=_axes[1])
    _axes[1].set_title("Highway (mpg)\ntrustworthy")
    sns.boxenplot(y=df2.highway_mpg, k_depth="tukey", ax=_axes[2])
    _axes[2].set_title("Highway (mpg)\ntukey")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Z-scores

    A z-score is `(value - mean) / standard deviation`. The mean and the standard deviation skip missing values. The lines on the chart sit at -3 and 3. The function returns the original values whose z-scores are outside that range.
    """)
    return


@app.cell
def _(plt, sns):
    def z_score_outliers(data):
        mean_x = data.mean()
        std_x = data.std()
        z_scores = (data - mean_x) / std_x
        grid = sns.displot(z_scores, kde=True, height=3, aspect=1.5)
        grid.ax.axvline(x=-3, ymin=0, ymax=0.95, color="orangered", linewidth=2)
        grid.ax.axvline(x=3, ymin=0, ymax=0.95, color="orangered", linewidth=2)
        z_outliers = z_scores[(z_scores < -3) | (z_scores > 3)]
        values = z_outliers * std_x + mean_x
        return grid.fig, values

    return (z_score_outliers,)


@app.cell
def _(df2, z_score_outliers):
    horsepower_figure, horsepower_z = z_score_outliers(df2.horsepower)
    horsepower_figure.axes[0].set_title("Horsepower")
    return horsepower_figure, horsepower_z


@app.cell
def _(horsepower_z):
    print(horsepower_z.tolist())
    return


@app.cell
def _(df2, z_score_outliers):
    city_figure, city_z = z_score_outliers(df2.city_mpg)
    city_figure.axes[0].set_title("City (mpg)")
    return city_figure, city_z


@app.cell
def _(city_z):
    print(city_z.tolist())
    return


@app.cell
def _(df2, z_score_outliers):
    highway_figure, highway_z = z_score_outliers(df2.highway_mpg)
    highway_figure.axes[0].set_title("Highway (mpg)")
    return highway_figure, highway_z


@app.cell
def _(highway_z):
    print(highway_z.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two rules do not mark the same points. For city fuel economy, the z-score rule marks 47, 49, and 45. The boxplot with `whis=1.5` marks 47 and 49. The value 45 is more than 3 standard deviations from the mean, and it is still inside the 1.5 IQR whiskers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plots

    A scatter plot shows two columns together. A point that is extreme in both, or extreme only as a pair, can be hard to see in the separate boxplots. `pairplot` draws every pair among the three columns.
    """)
    return


@app.cell
def _(df2, sns):
    pairs = sns.pairplot(data=df2)
    return pairs.fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Horsepower against city fuel economy falls as horsepower rises. The cars at the lower right sit far from that cloud: very high horsepower and low city fuel economy.
    """)
    return


@app.cell
def _(df2, sns):
    horsepower_city = sns.relplot(x=df2.horsepower, y=df2.city_mpg)
    horsepower_city.ax.set_title("Horsepower and city fuel economy")
    return horsepower_city.fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keeping rows with `horsepower < 250` removes 262 and 288. It also removes the two rows whose horsepower is missing, because a missing value is not less than 250. The remaining cloud has no point past 250.
    """)
    return


@app.cell
def _(df2):
    df3 = df2[df2.horsepower < 250]
    print("Rows kept =", len(df3))
    print("Highest horsepower kept =", df3.horsepower.max())
    return (df3,)


@app.cell
def _(df3, sns):
    horsepower_city_kept = sns.relplot(x=df3.horsepower, y=df3.city_mpg)
    horsepower_city_kept.ax.set_title("Horsepower below 250")
    return horsepower_city_kept.fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Keep the rows with horsepower below 200. Replace `None` with your code. Are the cars with horsepower 200 and 207 still in the table? What happens to the rows with a missing horsepower?
    """)
    return


@app.cell
def _():
    student_under_200 = None
    print(student_under_200)
    return (student_under_200,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_under_200 = df2[df2.horsepower < 200]
```
**No.** 200 and 207 fail `horsepower < 200`, and so do 262 and 288. The two rows with a missing horsepower are left out as well, because a missing value is not less than 200.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A boxplot marks points beyond its whiskers. `whis` is the number of IQRs past the quartiles. The default is 1.5. A larger `whis` marks fewer points. `whis=0` marks every point outside the box.
    - For horsepower, `whis=1.5` marks six values and `whis=2` marks 262 and 288. City and highway fuel economy have their own cutoffs. At `whis=2`, city fuel economy has no marked points.
    - A boxenplot draws several boxes into the tails. `k_depth` chooses how those boxes are spaced: `proportion`, `trustworthy`, or `tukey`.
    - A z-score uses the mean and the standard deviation. The function in this lesson flags values beyond 3 standard deviations. That set is not the same as the 1.5 IQR set. City fuel economy of 45 is in one set and not in the other.
    - A scatter plot shows a pair of columns. `horsepower < 250` removes the two highest cars and also the rows whose horsepower is missing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. In a boxplot, which points are drawn one by one?
    2. What does `whis=1.5` measure from the quartiles?
    3. What happens to the marked points when `whis` gets larger?
    4. What does `whis=0` mark?
    5. What cutoff does `z_score_outliers` use?
    6. Why can a city fuel economy of 45 be flagged by the z-score and not by `whis=1.5`?
    7. Which rows does `horsepower < 250` remove, besides the cars above 250?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Points beyond the whiskers:** The box and the whiskers summarize the rest.
2. **1.5 IQRs:** The whisker ends at the quartile plus or minus 1.5 times the interquartile range.
3. **Fewer points are marked:** The whiskers reach farther, so more points fall inside them.
4. **Every point outside the box:** There is no whisker. The box is the middle half of the data.
5. **3 standard deviations:** Values with a z-score below -3 or above 3 are returned.
6. **The cutoffs differ:** 45 is more than 3 standard deviations from the mean, and it is still inside the 1.5 IQR whiskers.
7. **The missing horsepower rows:** A missing value is not less than 250, so those two rows fail the filter.
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
