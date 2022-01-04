import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def getAnalysis(fields, filtering, ext, sep, title) -> list[dict[str, list]]:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    df.fillna(0)

    continentColumn = ""
    countryColumn = ""
    deathColumn = ""
    dateColumn = ""
    vaccineColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Continente":
            continentColumn = item["match"]
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Muertes":
            deathColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Vacunas":
            vaccineColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    # dateRow = df.iloc[0][dateColumn]
    df["JoinedDate"] = pd.to_datetime(df[dateColumn], infer_datetime_format=True)
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if title == "Análisis del número de muertes por coronavirus en un País":
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df))

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x[deathColumn],
            f"Análisis del número de muertes por coronavirus en {countryColumn}",
            "Muertes",
        )
        result = [
            {
                "pais": countryField,
                "rmse": pre[0],
                "r2": pre[1],
                "ecuacion": pre[2],
                "intercepto": pre[3],
                "coef": pre[4],
            }
        ]

        return result
    elif title == "Ánalisis Comparativo de Vacunación entre 2 paises":
        countryField1 = ""
        countryField2 = ""
        for filt in filtering:
            if filt["key"] == "Pais1":
                countryField1 = filt["value"]
            if filt["key"] == "Pais2":
                countryField2 = filt["value"]

        df1 = filterRows(df, countryColumn, countryField1)
        df1_ready = cleanRows(df1, dateColumn)
        df1_x = df1_ready[0]
        df1_x["Days"] = np.arange(len(df1))
        pre1 = predict(
            np.asarray(df1_x["Days"]).reshape(-1, 1),
            df1_x[deathColumn],
            f"Ánalisis Comparativo de Vacunación entre 2 paises",
            "Vacunación",
        )

        df2 = filterRows(df, countryColumn, countryField2)
        df2_ready = cleanRows(df2, dateColumn)
        df2_x = df2_ready[0]
        df2_x["Days"] = np.arange(len(df2))
        pre2 = predict(
            np.asarray(df1_x["Days"]).reshape(-1, 1),
            df1_x[deathColumn],
            f"Ánalisis Comparativo de Vacunación entre 2 paises",
            "Vacunación",
        )

        genGraph([pre1[5], pre2[5]])
        result = [{countryField1: pre1}, {countryField2: pre2}]

        return result
    else:  # Ánalisis Comparativo entres 2 o más paises o continentes
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]

        df = filterRows(df, countryColumn, countryColumn)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df_x))

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de mortalidad en {countryField}",
            "Muertes",
        )
        return pre


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
    plt.savefig("analysis.jpg")

    rmse = np.sqrt(mean_squared_error(y, y_))
    r2 = r2_score(y, y_)
    coef = regr.coef_
    intercept = regr.intercept_
    equation = f"y =  {coef[-1]} b + {intercept}"

    return [rmse, r2, equation, intercept, coef[-1], [x, y, y_]]


def genGraph(datas: list):
    # datas -> [ {title, y_label, x, y, y_} ]
    fig, ax = plt.subplots()

    for idx, data in datas:
        ax.scatter(data[0], data[1], color=colors[idx]["scatColor"], label="Casos")
        ax.plot(data[0], data[2], color=colors[idx]["plotColor"], label="Modelo casos")

    # ax.scatter(data1[0], data1[1], color=colors[-1]["scatColor"], label="Casos")
    # ax.plot(data1[0], data1[2], color=colors[-1]["plotColor"], label="Modelo casos")

    # ax.scatter(data2[0], data2[1], color=colors[0]["scatColor"], label="Muertes")
    # ax.plot(data2[0], data2[2], color=colors[0]["plotColor"], label="Modelo muertes")

    ax.legend()

    plt.title("Analisis comparativo de vacunación entre 2 países")
    plt.xlabel("Días")
    plt.ylabel("Vacunación")
    plt.savefig("analysis.jpg")
