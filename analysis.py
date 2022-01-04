import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def getAnalysis(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    continentColumn = ""
    countryColumn = ""
    dateColumn = ""
    totalColumn = ""
    confirmColumn = ""
    deathColumn = ""
    recoveryColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Continente":
            continentColumn = item["match"]
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Total":
            totalColumn = item["match"]
        if item["require"] == "Confirmados":
            confirmColumn = item["match"]
        if item["require"] == "Muertes":
            deathColumn = item["match"]
        if item["require"] == "Recuperados":
            recoveryColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    # dateRow = df.iloc[0][dateColumn]
    df["JoinedDate"] = pd.to_datetime(df[dateColumn], infer_datetime_format=True)
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if (
        title
        == "Tasa de comportamiento de casos activos en relación al número de muertes en un continente"
    ):
        continentField = ""
        for filt in filtering:
            if filt["key"] == "Continente":
                continentField = filt["value"]

        df = filterRows(df, continentColumn, continentField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df))

        df_x["Tasa"] = df_x[confirmColumn] - df_x[deathColumn]

        tasa = df_x[confirmColumn].sum() - df_x[deathColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de crecimiento de casos activos en relación al númeor de muertes en {continentField}",
            "Infectados",
        )
        return [pre[0], pre[1], pre[2], pre[3], pre[4], tasa]
    elif (
        title
        == "Tasa de crecimiento de casos en relación con nuevos casos diarios y tasa de muerte"
    ):
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df_x))

        df_x["Tasa"] = df_x[confirmColumn] - df_x[deathColumn] - df_x[recoveryColumn]

        tasa = df_x[confirmColumn].sum() - df_x[deathColumn].sum()
        tasa = tasa - df_x[recoveryColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de crecimiento de casos en relación de casos diarios y tasa de muerte",
            "Infectados",
        )
        return [pre[0], pre[1], pre[2], pre[3], pre[4], tasa]
    else:  # Tasa de mortalidad por coronavirus (COVID-19) en un país
        continentField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                continentField = filt["value"]

        df = filterRows(df, countryColumn, countryColumn)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df_x))

        df_x["Tasa"] = df_x[totalColumn] - df_x[deathColumn]

        tasa = df_x[totalColumn].sum() - df_x[deathColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de mortalidad en {continentField}",
            "Muertes",
        )
        return [pre[0], pre[1], pre[2], pre[3], pre[4], tasa]


def filterRows(dataFrame: DataFrame, columnName: str, rowName: str) -> DataFrame:
    print(f"filtrar {columnName} por {rowName}")
    dataFrame = dataFrame.loc[lambda x: x[columnName] == rowName]
    return dataFrame


def cleanRows(dataFrame: DataFrame, date):
    df_cleaned = dataFrame.drop_duplicates("JoinedDate")
    df_summed = dataFrame.groupby(by=["Year", "Month", date]).sum()
    return [df_cleaned, df_summed]


def predict(x, y, title: str, y_label: str) -> list:
    regr = LinearRegression()
    regr.fit(x, y)
    y_ = regr.predict(x)

    # GRAFICA
    indexColor = 2
    if y_label == "Muertes":
        indexColor = 0
    elif y_label == "Confirmados" or y_label == "Infectados":
        indexColor = -1
    else:
        indexColor = 2

    # title, y_label, x, y, y_
    fig, ax = plt.subplots()

    ax.scatter(x, y, color=colors[indexColor]["scatColor"], label="Muestra")
    ax.plot(x, y_, color=colors[indexColor]["plotColor"], label="Tendencia")
    ax.legend()

    plt.title(title)
    plt.xlabel("Días")
    plt.ylabel(y_label)
    plt.savefig("rate.jpg")

    rmse = np.sqrt(mean_squared_error(y, y_))
    r2 = r2_score(y, y_)
    coef = regr.coef_
    intercept = regr.intercept_
    equation = f"y =  {coef[-1]} b + {intercept}"

    return [rmse, r2, equation, intercept, coef[-1]]
