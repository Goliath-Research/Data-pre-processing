# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
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
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Categorical Features

    ## Objectives

    - Turn a category with no order into integers with label encoding and with Pandas category codes.
    - Turn the same category into 0/1 columns with one-hot encoding, and see what `drop='first'` leaves out.
    - Turn a category that has a real order into integers with a dictionary and `map`.

    ## Background

    A column such as body style is stored as text. Most numeric methods cannot use that text until each category has a number. The number should match the meaning of the category. Body style has no order, so a single code invents an order that is not in the data. One-hot encoding avoids that by giving each style its own 0/1 column. Buying price does have an order, from low to very high, so an ordered code keeps that order.

    This lesson does not fit a model, so it does not measure a change in model accuracy. It shows what each encoding puts in the table.

    ## Datasets Used

    - **Automobile Dataset** from the UCI Machine Learning Repository. The example column is `body_style`.
    - **Car Evaluation Dataset** from the UCI Machine Learning Repository. `buying`, `maint`, `lug_boot`, and `safety` have an order.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the files are read from

    Pandas can read a file directly from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can download a file from the UCI website, and it can read `car.data` from this folder.
    - **Inside a web browser.** This notebook can run inside a web page. The browser only lets a page download files from websites that allow it. This security rule is called **CORS**. The UCI website does not allow it, so the browser blocks that download.

    This lesson reads GitHub copies of both files. GitHub allows a browser page to read its files, so the copies work in both places. The UCI addresses and the local file name are kept in comments.
    """)
    return


@app.cell
def _():
    auto_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Introduction-to-Data-Science/main/"
        "04-Introduction%20to%20Pandas/public/imports-85.data"
    )
    car_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Data-pre-processing/master/Feature-Engineering/car.data"
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
    return auto_file, car_file, headers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automobile dataset

    The file has no header row. `na_values='?'` reads question marks as missing. The table has 26 columns, so the first rows are shown as HTML.
    """)
    return


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


@app.cell
def _(df):
    print(df.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `body_style` is text. Five styles appear. Sedan is the most common, and convertible is the least common.
    """)
    return


@app.cell
def _(df):
    print(df.body_style.value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label encoding

    `LabelEncoder` gives each different text one integer. The integers follow alphabetical order: convertible is 0, hardtop is 1, hatchback is 2, sedan is 3, and wagon is 4. Nothing in the data says that a wagon is "more" than a sedan. The numbers are only labels.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder

    return (LabelEncoder,)


@app.cell
def _(LabelEncoder, df):
    df_label = df[["body_style"]].copy()
    df_label["body_style_n"] = LabelEncoder().fit_transform(df_label["body_style"])
    print(df_label.head())
    return (df_label,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One row of each style shows the code. Sorting by the code shows the alphabetical order.
    """)
    return


@app.cell
def _(df_label):
    style_codes = (
        df_label[["body_style", "body_style_n"]]
        .drop_duplicates()
        .sort_values("body_style_n")
    )
    print(style_codes)
    return (style_codes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Category codes

    `astype('category')` stores the same texts as a Pandas category. The categories are sorted alphabetically, and `.cat.codes` numbers them from 0. Those codes match `LabelEncoder`.
    """)
    return


@app.cell
def _(df_label):
    df_codes = df_label.copy()
    df_codes["body_style"] = df_codes["body_style"].astype("category")
    df_codes["body_style_cat"] = df_codes["body_style"].cat.codes
    print(df_codes.dtypes)
    return (df_codes,)


@app.cell
def _(df_codes):
    code_match = (
        df_codes[["body_style", "body_style_n", "body_style_cat"]]
        .drop_duplicates()
        .sort_values("body_style_n")
    )
    print(code_match)
    print("Same codes:", bool((df_codes.body_style_n == df_codes.body_style_cat).all()))
    return (code_match,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The cell prints the code for sedan. Change `'sedan'` to `'wagon'` and run the cell again. What code does wagon have?
    """)
    return


@app.cell
def _(df_label):
    style_name = "sedan"
    style_code = df_label.loc[
        df_label.body_style == style_name, "body_style_n"
    ].iloc[0]
    print(style_name, int(style_code))
    return style_code, style_name


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
style_name = "wagon"
```
**4.** The codes are alphabetical. Wagon is the last of the five styles, so its code is 4. Sedan is 3.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-hot encoding

    One-hot encoding makes one 0/1 column per style. A 1 means the row has that style. The other columns in that row are 0. No column is "larger" than another, so the encoding does not invent an order.

    The new table has 6 columns, so it is shown as HTML. The names `0` through `4` follow the same alphabetical order as the label codes.
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import OneHotEncoder

    return (OneHotEncoder,)


@app.cell
def _(OneHotEncoder, df_label, mo, pd):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df_label[["body_style"]])
    dfo = pd.DataFrame(encoded)
    dfo["body_style"] = df_label["body_style"].to_numpy()
    mo.Html(dfo.head(8).to_html())
    return dfo, encoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `drop='first'` leaves out the first category, convertible. A convertible row is then 0 in every remaining column. The other four styles still have their own column. A later linear model can use this shorter table, because the dropped column is determined by the other four: if they are all 0, the style is convertible.
    """)
    return


@app.cell
def _(OneHotEncoder, df_label, mo, pd):
    encoder_drop = OneHotEncoder(sparse_output=False, drop="first")
    encoded_drop = encoder_drop.fit_transform(df_label[["body_style"]])
    dfo_drop = pd.DataFrame(encoded_drop)
    dfo_drop["body_style"] = df_label["body_style"].to_numpy()
    mo.Html(dfo_drop.head(8).to_html())
    return dfo_drop, encoder_drop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pd.get_dummies` builds the same 0/1 columns and names them. `dtype=int` stores 0 and 1. Without it, current Pandas stores `True` and `False`. `drop_first=True` again drops convertible.
    """)
    return


@app.cell
def _(df_label, mo, pd):
    dummies = pd.get_dummies(df_label.body_style, prefix="bs", dtype=int)
    dummies["body_style"] = df_label["body_style"].to_numpy()
    mo.Html(dummies.head(8).to_html())
    return (dummies,)


@app.cell
def _(df_label, mo, pd):
    dummies_drop = pd.get_dummies(
        df_label.body_style, prefix="bs", drop_first=True, dtype=int
    )
    dummies_drop["body_style"] = df_label["body_style"].to_numpy()
    mo.Html(dummies_drop.head(8).to_html())
    return (dummies_drop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Build 0/1 columns for `fuel_type`, with the prefix `fuel`, and keep both categories. Replace `None` with your code. How many 0/1 columns do you get?
    """)
    return


@app.cell
def _():
    student_fuel = None
    print(student_fuel)
    return (student_fuel,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_fuel = pd.get_dummies(df.fuel_type, prefix="fuel", dtype=int)
```
**Two.** The categories are diesel and gas, so the columns are `fuel_diesel` and `fuel_gas`.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinal encoding

    The Car Evaluation file has 1,728 rows and no header. `buying` and `maint` use the same four words, and those words have an order: low, medium, high, very high. A dictionary can write that order as 0, 1, 2, and 3. Label encoding would sort the words alphabetically and would put `high` before `low`.
    """)
    return


@app.cell
def _(car_file, mo, pd):
    car_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "carEval"]
    # Reading from the UCI website (works when Python runs on your computer):
    # car = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    #     header=None,
    #     names=car_columns,
    # )
    # On this computer the file is also next to the notebook:
    # car = pd.read_csv("car.data", header=None, names=car_columns)
    car = pd.read_csv(car_file, header=None, names=car_columns)
    mo.Html(car.head().to_html())
    return (car,)


@app.cell
def _(car):
    print(list(car.buying.unique()))
    print(list(car.maint.unique()))
    return


@app.cell
def _():
    price_dict = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
    return (price_dict,)


@app.cell
def _(car, price_dict):
    buying_codes = car[["buying"]].copy()
    buying_codes["buying_cat"] = buying_codes.buying.map(price_dict)
    print(buying_codes.drop_duplicates().sort_values("buying_cat"))
    return (buying_codes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same dictionary is applied to `buying` and to `maint`. The other columns stay as text for now. This table has 7 columns, so it is shown as HTML.
    """)
    return


@app.cell
def _(car, mo, price_dict):
    car_price = car.copy()
    car_price["buying"] = car_price.buying.map(price_dict)
    car_price["maint"] = car_price.maint.map(price_dict)
    mo.Html(car_price.head().to_html())
    return (car_price,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `lug_boot` is small, medium, or big. `safety` is low, medium, or high. Each dictionary follows that order.
    """)
    return


@app.cell
def _(car_price):
    print(car_price.lug_boot.value_counts())
    return


@app.cell
def _(car_price):
    boot_dict = {"small": 0, "med": 1, "big": 2}
    car_boot = car_price.copy()
    car_boot["lug_boot"] = car_boot.lug_boot.map(boot_dict)
    print(car_boot[["lug_boot"]].drop_duplicates().sort_values("lug_boot"))
    return boot_dict, car_boot


@app.cell
def _(car_boot):
    print(car_boot.safety.value_counts())
    return


@app.cell
def _(car_boot, mo):
    safety_dict = {"low": 0, "med": 1, "high": 2}
    car_safety = car_boot.copy()
    car_safety["safety"] = car_safety.safety.map(safety_dict)
    mo.Html(car_safety.head().to_html())
    return car_safety, safety_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `doors`, `persons`, and `carEval` are still text. `doors` and `persons` also have an order. `carEval` is the evaluation to predict, and this lesson leaves it as text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Map `doors` with `{'2': 2, '3': 3, '4': 4, '5more': 5}`. Replace `None` with your code. What number does `5more` become?
    """)
    return


@app.cell
def _():
    student_doors = None
    print(student_doors)
    return (student_doors,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
door_dict = {"2": 2, "3": 3, "4": 4, "5more": 5}
student_doors = car_safety.copy()
student_doors["doors"] = student_doors["doors"].map(door_dict)
```
**5.** `5more` means five or more doors, so the code keeps it after 4. A label encoder would sort the text and would not use these numbers.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - `LabelEncoder` and `.cat.codes` give `body_style` the integers 0 through 4 in alphabetical order. The two methods match. The integers do not mean that one body style is larger than another.
    - One-hot encoding makes one 0/1 column per style. `drop='first'` and `drop_first=True` omit convertible. A row of zeros then means convertible.
    - `pd.get_dummies` names those columns. `dtype=int` stores 0 and 1.
    - `buying`, `maint`, `lug_boot`, and `safety` have a real order. A dictionary and `map` write that order. Alphabetical label encoding would put `high` before `low`.
    - `doors`, `persons`, and `carEval` are still text at the end of the lesson.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why is a single integer a poor summary of body style?
    2. What integer does label encoding give sedan?
    3. Do `.cat.codes` and `LabelEncoder` give `body_style` the same integers?
    4. What does a 1 mean in a one-hot column?
    5. Which body style becomes a row of zeros when the first category is dropped?
    6. Why is a dictionary a better encoding than label encoding for `buying`?
    7. Which three Car Evaluation columns are still text after this lesson?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **The styles have no order:** The integers 0 through 4 look like a scale, and body style is not a scale.
2. **3:** The codes are alphabetical. Sedan comes after convertible, hardtop, and hatchback.
3. **Yes:** Both sort the five styles alphabetically and number them from 0.
4. **The row has that category:** The other one-hot columns in that row are 0.
5. **Convertible:** It is the first style in alphabetical order, so `drop='first'` leaves it out.
6. **The words have an order:** Low, medium, high, and very high become 0, 1, 2, and 3. Alphabetical order would put `high` before `low`.
7. **`doors`, `persons`, and `carEval`:** The lesson encodes buying price, maintenance, luggage boot, and safety.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    - [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    """)
    return


if __name__ == "__main__":
    app.run()
