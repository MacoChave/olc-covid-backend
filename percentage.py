import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def getPercentage(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    continentColumn = ""
    regionColumn = ""
    countryColumn = ""
    confirmColumn = ""
    dateColumn = ""
    genreColumn = ""
    totalColumn = ""
    deathsColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Continente":
            continentColumn = item["match"]
        if item["require"] == "Region":
            regionColumn = item["match"]
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Confirmados":
            confirmColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Género":
            genreColumn = item["match"]
        if item["require"] == "Total":
            totalColumn = item["match"]
        if item["require"] == "Muertes":
            deathsColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    # dateRow = df.iloc[0][dateColumn]
    df["JoinedDate"] = pd.to_datetime(df[dateColumn], infer_datetime_format=True)
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if (
        title
        == "Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo"
    ):
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]

        df = filterRows(df, countryColumn, countryField)

        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        df_x["Days"] = np.arange(len(df))

        df_x["Percentage"] = df_x[genreColumn] / df_x[confirmColumn]
        df_x = df_x.fillna(0)
        df_x["Percentage"] = df_x["Percentage"] * 100

        total = df_x[genreColumn].sum()
        confirm = df_x[confirmColumn].sum()
        percent = (total * 100) / confirm

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Percentage"],
            f"Porcentaje de hombres infectados en {countryField} desde el 1er caso",
            "Infectados",
        )
        return [pre[0], pre[1], pre[2], pre[3], pre[4], percent]
    else:  # Porcentaje de muertes frente al total de casos en un país, región o continente
        countryField = ""
        regionField = ""
        continentField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]
            if filt["key"] == "Region":
                regionField = filt["value"]
            if filt["key"] == "Continente":
                continentField = filt["value"]

        lugar = ""
        if countryColumn == "":
            df = filterRows(df, countryColumn, countryField)
            lugar = f"país {countryField}"
        if continentColumn == "":
            df = filterRows(df, continentColumn, continentField)
            lugar = f"continente {continentField}"
        if regionColumn == "":
            df = filterRows(df, regionColumn, regionField)
            lugar = f"región {regionField}"

        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        df_x["Days"] = np.arange(len(df_x))

        df_x["Percentage"] = df_x[deathsColumn] / df_x[totalColumn]
        df_x = df_x.fillna(0)
        df_x["Percentage"] = df_x["Percentage"] * 100
        df_x["Days"] = np.arange(len(df))

        total = df_x[deathsColumn].sum()
        confirm = df_x[totalColumn].sum()
        percent = (total * 100) / confirm

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[totalColumn],
            f"Porcentaje de muertes sobre el total de casos en {lugar}",
            "Muertes",
        )
        return [pre[0], pre[1], pre[2], pre[3], pre[4], percent]


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
    plt.savefig("percentage.jpg")

    rmse = np.sqrt(mean_squared_error(y, y_))
    r2 = r2_score(y, y_)
    coef = regr.coef_
    intercept = regr.intercept_
    equation = f"y =  {coef[-1]} b + {intercept}"

    return [rmse, r2, equation, intercept, coef[-1]]
