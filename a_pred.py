# predictiin my way downtown

import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import messagebox


# loady load

folder_path = "/Users/kelly/Downloads/artwork_details"

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

print("Loaded rows:", len(df))


# so many uneedded columns

df = df[
    [
        "artist",
        "description",
        "purchase_price",
        "sale_price",
        "gross_appreciation_multiplier",
        "gross_appreciation_period",
    ]
]


# sweep sweep sweep

def clean_money(value):

    if pd.isna(value):
        return np.nan

    value = str(value).replace("$", "").replace(",", "").strip()

    # kkip accidental header rows
    if value.lower() == "purchase_price":
        return np.nan

    multiplier = 1

    if "K" in value:
        multiplier = 1_000
        value = value.replace("K", "")

    elif "M" in value:
        multiplier = 1_000_000
        value = value.replace("M", "")

    elif "B" in value:
        multiplier = 1_000_000_000
        value = value.replace("B", "")

    try:
        return float(value) * multiplier

    except:
        return np.nan


# converts:
def clean_multiplier(value):

    if pd.isna(value):
        return np.nan

    value = (
        str(value)
        .replace("x", "")
        .replace(",", "")
        .strip()
    )

    # skip accidental header rows
    if value.lower() == "gross_appreciation_multiplier":
        return np.nan

    try:
        return float(value)

    except:
        return np.nan

# converts:
def clean_years(value):

    if pd.isna(value):
        return np.nan

    value = (
        str(value)
        .replace("years", "")
        .replace("year", "")
        .replace(",", "")
        .strip()
    )

    # skip accidental header rows
    if value.lower() == "gross_appreciation_period":
        return np.nan

    try:
        return float(value)

    except:
        return np.nan

# remove the baddd ones
df = df.dropna(
    subset=[
        "purchase_price",
        "sale_price",
    ]
)

# sweep sweep sweep

df["purchase_price"] = df["purchase_price"].apply(clean_money)

df["sale_price"] = df["sale_price"].apply(clean_money)

df["gross_appreciation_multiplier"] = (
    df["gross_appreciation_multiplier"]
    .apply(clean_multiplier)
)

df["gross_appreciation_period"] = (
    df["gross_appreciation_period"]
    .apply(clean_years)
)


# missing my data?  no problem

df["artist"] = df["artist"].fillna("Unknown Artist")

df["description"] = df["description"].fillna(
    "Unknown Description"
)

#remove rows with no price
df = df.dropna(subset=["sale_price"])

print("\nCleaned dataset preview:\n")
print(df.head())


# features and target and stuff

X = df[
    [
        "artist",
        "description",
        "purchase_price",
        "gross_appreciation_multiplier",
        "gross_appreciation_period",
    ]
]

y = df["sale_price"]

# processin

# description text processing
# lets model understand similar phrases
description_transformer = TfidfVectorizer(
    stop_words="english"
)

# names
artist_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# numeric features
numeric_features = [
    "purchase_price",
    "gross_appreciation_multiplier",
    "gross_appreciation_period",
]

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

# combine all preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "description",
            description_transformer,
            "description",
        ),

        (
            "artist",
            artist_transformer,
            ["artist"],
        ),

        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
    ]
)

# teaching rocks how to think time

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",

            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=4,
                random_state=42,
            ),
        ),
    ]
)


# testing the rocks time


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining model...\n")

model.fit(X_train, y_train)



# grading exam papers


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("====== MODEL PERFORMANCE ======")

print("Mean Absolute Error: ${:,.2f}".format(mae))

print("R^2 Score:", round(r2, 3))


# gui cuz termial hard to use


def predict_price():

    try:

        # required
        buy_price_text = buy_price_entry.get().strip()
        years_text = years_entry.get().strip()

        # check
        if buy_price_text == "" or years_text == "":
            messagebox.showerror(
                "Missing Required Fields",
                "Buy Price and Years Passed are required."
            )
            return

        # optional
        artist_input = artist_entry.get().strip()
        description_input = description_entry.get().strip()
        multiplier_text = multiplier_entry.get().strip()

        # default vals
        if artist_input == "":
            artist_input = "Unknown Artist"

        if description_input == "":
            description_input = "Unknown Description"

        # required
        buy_price_input = float(buy_price_text)
        years_input = float(years_text)

        # optional
        if multiplier_text == "":
            multiplier_input = (
                df[
                    "gross_appreciation_multiplier"
                ].median()
            )
        else:
            multiplier_input = float(multiplier_text)

        # create dataframe
        user_data = pd.DataFrame(
            {
                "artist": [artist_input],

                "description": [description_input],

                "purchase_price": [buy_price_input],

                "gross_appreciation_multiplier": [
                    multiplier_input
                ],

                "gross_appreciation_period": [
                    years_input
                ],
            }
        )

        # prredict
        predicted_price = model.predict(user_data)[0]

        # display result
        result_label.config(
            text=(
                "Predicted Sell Price:\n"
                f"${predicted_price:,.2f}"
            )
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers."
        )


# setup

root = tk.Tk()

root.title("Famous Artwork Price Predictor")

root.geometry("500x500")

root.configure(bg="#f2f2f2")

# title

header = tk.Label(
    root,
    text="Famous Artwork Price Predictor",
    font=("Arial", 20, "bold"),
    bg="#f2f2f2",
)

header.pack(pady=20)



# input


def make_label(text):
    label = tk.Label(
        root,
        text=text,
        font=("Arial", 11),
        bg="#f2f2f2",
    )
    label.pack()


# Artist
make_label("Artist Name (optional)")
artist_entry = tk.Entry(root, width=40)
artist_entry.pack(pady=5)


# Description
make_label("Description (optional)")
description_entry = tk.Entry(root, width=40)
description_entry.pack(pady=5)


# Buy Price
make_label("Buy Price (required)")
buy_price_entry = tk.Entry(root, width=40)
buy_price_entry.pack(pady=5)


# Years Passed
make_label("Years Passed (required)")
years_entry = tk.Entry(root, width=40)
years_entry.pack(pady=5)


# Multiplier
make_label("Appreciation Multiplier (optional)")
multiplier_entry = tk.Entry(root, width=40)
multiplier_entry.pack(pady=5)


# predict that stuff


predict_button = tk.Button(
    root,
    text="Predict",
    font=("Arial", 14, "bold"),
    padx=20,
    pady=10,
    command=predict_price,
)

predict_button.pack(pady=25)


# tadaaa results

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#f2f2f2",
)

result_label.pack(pady=20)


# startup

root.mainloop()
