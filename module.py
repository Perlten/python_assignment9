import tensorflow as tf
import urllib.request as request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Part 1


def download_data():
    url = "https://raw.githubusercontent.com/PranayMalhotra/NBA-player-career-lengths/master/nba_logreg.csv"
    request.urlretrieve(url, "nba_data.csv")
    return pd.read_csv("nba_data.csv",)

# Part 2


def print_scatter_on_gp(df):
    plt.scatter(df["gp"], df["target_5yrs"])
    plt.show()


def print_scatter_on_min(df):
    plt.scatter(df["min"], df["target_5yrs"])
    plt.show()


def create_model(df):
    y = np.array(df["target_5yrs"]).flatten()
    x = df.iloc[:, 1:].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
   
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_acc)

if __name__ == "__main__":
    df = download_data()
    # print(df.head(10))
    # print_scatter_on_min(df)
    # sorted_df = df.sort_values("3p_made", ascending=False)
    # print(sorted_df["name"][:10])
    create_model(df)
