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
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Extraction with Principal Component Analysis

    ## Objectives

    - Standardize the numeric automobile predictors and compare their correlation before and after.
    - Fit a principal component analysis and read the variance each component explains.
    - Keep enough components to explain 80 percent of the variance, and see how the original columns combine into those components.

    ## Background

    Feature extraction builds new columns from the old ones. Principal component analysis (PCA) builds them so that the first component carries as much variance as it can, and later components are uncorrelated with the ones before them. Each new column is a weighted sum of the original columns.

    This lesson keeps the components that together explain 80 percent of the variance. It does not fit a prediction model, so it does not measure a change in accuracy or speed.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. Rows with a missing value are removed. `price` is set aside for the scatter plot. The other 15 numeric columns are the inputs to PCA.
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
def _(auto_file, headers, mo, pd):
    # df_raw = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    #     header=None,
    #     names=headers,
    #     na_values="?",
    # )
    df_raw = pd.read_csv(auto_file, header=None, names=headers, na_values="?")
    mo.Html(df_raw.head(3).to_html())
    return (df_raw,)


@app.cell
def _(df_raw):
    print(df_raw.isnull().sum())
    return


@app.cell
def _(df_raw):
    df = df_raw.dropna()
    print(df.isnull().sum().sum())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The numeric columns have dtype `int64` or `float64`. There are 16 of them, including `price`. PCA uses the other 15. `price` is kept only so the last scatter plot can color the cars.
    """)
    return


@app.cell
def _(df):
    numeric_columns = [
        column
        for column in df.columns
        if df[column].dtype == "float64" or df[column].dtype == "int64"
    ]
    predictors = df[numeric_columns].drop(columns="price")
    price = df.price.to_numpy()
    print(list(predictors.columns))
    return predictors, price


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation

    Several predictors move together. `length`, `width`, `curb_weight`, and `engine_size` are strongly correlated, and the two miles-per-gallon columns are strongly correlated with each other. A heatmap of all 15 is the picture of that overlap.
    """)
    return


@app.cell
def _(plt, predictors, sns):
    _figure, _axes = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        predictors.corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        linewidths=0.5,
        ax=_axes,
    )
    _axes.set_title("Correlation of the original predictors")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Standardizing

    PCA uses variances. A column such as `curb_weight` has a much larger spread than `compression_ratio`, so it would dominate the first component if the columns kept their original units. `StandardScaler` subtracts each column's mean and divides by its standard deviation. After that, every column has mean 0 and standard deviation 1.

    The box plot of the original columns uses their own units. The box plot of the standardized columns uses a shared scale.
    """)
    return


@app.cell
def _(predictors):
    _axes = predictors.boxplot(figsize=(14, 5), color="blue")
    _axes.set_title("Original predictors")
    _axes.tick_params(axis="x", labelrotation=45)
    return _axes.figure


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler,)


@app.cell
def _(StandardScaler, mo, pd, predictors):
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(predictors),
        columns=predictors.columns,
    )
    mo.Html(scaled.head(3).to_html())
    return scaled, scaler


@app.cell
def _(scaled):
    _axes = scaled.boxplot(figsize=(14, 5), color="green")
    _axes.set_title("Standardized predictors")
    _axes.tick_params(axis="x", labelrotation=45)
    return _axes.figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Standardizing does not change the Pearson correlations. Each column was shifted and stretched, and those two steps do not change how two columns move together. The heatmap matches the one above.
    """)
    return


@app.cell
def _(plt, scaled, sns):
    _figure, _axes = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        scaled.corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        linewidths=0.5,
        ax=_axes,
    )
    _axes.set_title("Correlation of the standardized predictors")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principal components

    `PCA()` with no `n_components` builds one component for every predictor. The transformed table still has 15 columns. `explained_variance_ratio_` is the share of variance in each component. Multiplying by 100 turns that share into a percent.

    The first component explains 47.29 percent. The first two together explain 64.40 percent. The first four together explain 81.41 percent.
    """)
    return


@app.cell
def _():
    from sklearn.decomposition import PCA

    return (PCA,)


@app.cell
def _(PCA, scaled):
    pca = PCA()
    pca.fit(scaled)
    print("Original shape:", scaled.shape)
    print("Transformed shape:", pca.transform(scaled).shape)
    return (pca,)


@app.cell
def _(np, pca):
    component_number = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for _number, _ratio, _total in zip(component_number, pca.explained_variance_ratio_, cumulative):
        print(
            f"Component {_number:2d}    variance {_ratio * 100:6.2f}%    cumulative {_total * 100:6.2f}%"
        )
    return component_number, cumulative


@app.cell
def _(component_number, cumulative, pca, plt):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(component_number, cumulative, "o", color="firebrick")
    _axes[0].axvline(4, color="firebrick")
    _axes[0].set_ylim(-0.1, 1.1)
    _axes[0].set_xlabel("Number of components")
    _axes[0].set_ylabel("Cumulative explained variance")
    _axes[0].set_title("Cumulative variance")
    _axes[1].plot(component_number, pca.explained_variance_ratio_, "o", color="darkgreen")
    _axes[1].axvline(4, color="darkgreen")
    _axes[1].set_ylim(-0.1, 1.1)
    _axes[1].set_xlabel("Component")
    _axes[1].set_ylabel("Explained variance ratio")
    _axes[1].set_title("Variance of each component")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The vertical line is at four components. Does the cumulative variance at that line pass 80 percent? Replace `None` with `True` or `False`.
    """)
    return


@app.cell
def _():
    student_four_pass_80 = None
    print(student_four_pass_80)
    return (student_four_pass_80,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_four_pass_80 = True
```
**True.** Four components explain 81.41 percent. Three components explain 74.70 percent, which is still under 80.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keeping 80 percent of the variance

    `n_components=0.8` tells PCA to keep the smallest number of components whose ratios add up to at least 0.80. That number is 4. `whiten=True` scales those four columns so each has variance 1. They stay uncorrelated.
    """)
    return


@app.cell
def _(PCA, pd, scaled):
    pca_80 = PCA(n_components=0.8, whiten=True)
    factors = pd.DataFrame(
        pca_80.fit_transform(scaled),
        columns=["f1", "f2", "f3", "f4"],
    )
    print("Original shape:", scaled.shape)
    print("Transformed shape:", factors.shape)
    return factors, pca_80


@app.cell
def _(factors, pca_80):
    ratios = pca_80.explained_variance_ratio_
    for _index, _ratio in enumerate(ratios, start=1):
        print(f"Factor {_index}    {_ratio * 100:.2f}%")
    print(f"Cumulative    {ratios.sum() * 100:.2f}%")
    return (ratios,)


@app.cell
def _(factors):
    _axes = factors.boxplot(figsize=(8, 4), color="firebrick")
    _axes.set_title("Four components")
    return _axes.figure


@app.cell
def _(factors):
    print(factors.describe().round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each factor has a mean near 0 and a standard deviation near 1. The correlations among `f1`, `f2`, `f3`, and `f4` are 0, apart from tiny rounding error.
    """)
    return


@app.cell
def _(factors):
    print(factors.corr().round(6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the original columns enter the components

    `components_` holds the weights. The table has one row per original predictor and one column per factor. A large weight, positive or negative, means that predictor pulls that factor. The heatmap is the same table.
    """)
    return


@app.cell
def _(pca_80, pd):
    loadings = pd.DataFrame(
        pca_80.components_.T,
        index=pca_80.feature_names_in_,
        columns=["f1", "f2", "f3", "f4"],
    )
    print(loadings.round(3))
    return (loadings,)


@app.cell
def _(loadings, plt, sns):
    _figure, _axes = plt.subplots(figsize=(4, 8))
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=1,
        annot_kws={"size": 9},
        ax=_axes,
    )
    _axes.set_title("Weights")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two components and price

    The first two components explain 64.40 percent of the variance. The scatter plot puts each car on those two axes. Color is the car's `price`, which was not one of the 15 inputs.
    """)
    return


@app.cell
def _(factors, plt, price):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    _points = _axes.scatter(factors.f1, factors.f2, c=price, cmap="cool", s=30)
    _figure.colorbar(_points, ax=_axes, label="price")
    _axes.set_xlabel("Component 1")
    _axes.set_ylabel("Component 2")
    _axes.set_title("Cars on the first two components")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What percent of the variance do the first two components explain? Replace `None` with the number, without a percent sign.
    """)
    return


@app.cell
def _():
    student_two_components = None
    print(student_two_components)
    return (student_two_components,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_two_components = 64.40
```
**64.40.** The first component is 47.29 percent and the second is 17.11 percent.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - After dropping incomplete rows, PCA uses 15 numeric predictors. `price` is not one of them.
    - Standardizing gives every predictor mean 0 and standard deviation 1. The Pearson correlations stay the same.
    - The first component explains 47.29 percent of the variance. Two components explain 64.40 percent. Four components explain 81.41 percent.
    - `PCA(n_components=0.8, whiten=True)` keeps those four components and scales each one to variance 1. The four columns are uncorrelated.
    - Each component is a weighted sum of the 15 predictors. The loadings table shows the weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. How many numeric predictors go into the PCA?
    2. Why are the predictors standardized first?
    3. What happens to the Pearson correlations when the columns are standardized?
    4. How much variance does the first component explain?
    5. How many components does `n_components=0.8` keep?
    6. What does `whiten=True` do to those components?
    7. Was `price` one of the columns used to build the components?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **15:** There are 16 numeric columns. `price` is set aside.
2. **So one unit does not dominate:** `curb_weight` would outweigh a column with a small spread if the original units were kept.
3. **They stay the same:** Subtracting a mean and dividing by a standard deviation does not change the Pearson correlation.
4. **47.29 percent:** The second component adds 17.11 percent, for 64.40 percent together.
5. **Four:** Three components reach 74.70 percent. Four reach 81.41 percent.
6. **It scales them to variance 1:** The columns stay uncorrelated, with means near 0.
7. **No:** `price` only colors the scatter plot of the first two components.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    """)
    return


if __name__ == "__main__":
    app.run()
