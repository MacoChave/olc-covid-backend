import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def getTrend(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    countryColumn = ""
    confirmColumn = ""
    dateColumn = ""
    deptoColumn = ""
    vaccineColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Confirmados":
            confirmColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Departamento":
            deptoColumn = item["match"]
        if item["require"] == "Vacunación":
            vaccineColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    # dateRow = df.iloc[0][dateColumn]
    df["JoinedDate"] = pd.to_datetime(df[dateColumn], infer_datetime_format=True)
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if title == "Tendencia de la infección por Covid-19 en un País":
        daysField = ""
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]
            elif filt["key"] == "Dias":
                daysField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        df_x["Days"] = np.arange(len(df))
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysField,
            f"Tendencia de infectados en {countryField}",
            "Infectados",
        )
        return pre
    elif title == "Tendencia del número de infectados por día de un País":
        daysField = ""
        countryField = ""
        deptoField = ""
        for filt in filtering:
            if filt["key"] == "Dias":
                daysField = filt["value"]
            elif filt["key"] == "Pais":
                countryField = filt["value"]
            elif filt["key"] == "Departamento":
                deptoField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df = filterRows(df, deptoColumn, deptoField)
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            daysField,
            f"Tendenciade infectados en {countryField}",
            "Infectados",
        )
        return pre
    elif title == "Tendencia de la vacunación de un País":
        daysField = ""
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]
            elif filt["key"] == "Dias":
                daysField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            daysField,
            f"Tendencia de vacunación en {countryField}",
            "Vacunación",
        )
        return pre
    elif title == "Predicción de casos de un país para un año":
        yearField = ""
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]
            elif filt["key"] == "Año":
                yearField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        daysPredicted = int(yearField) * 365
        print(f"Days predicted: {daysPredicted}")
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysPredicted,
            f"Predicción de mortalidad en {countryField} para {yearField} años",
            "Confirmados",
        )
        return pre
    else:  # "Tendencia de casos confirmados de Coronavirus en un departamento de un País"
        countryField = ""
        deptoField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]
            if filt["key"] == "Departamento":
                deptoField = filt["value"]

        df = filterRows(df, countryColumn, countryColumn)
        df = filterRows(df, deptoField, deptoField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        df_x["Days"] = np.arange(len(df_x))
        # TODO: predictToEndYear
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            365,
            f"Tendencia de casos confirmados en {deptoField}, {countryField}",
            "Confirmados",
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


def predict(x, y, daysPredicted: int, title: str, y_label: str) -> list:
    pf = PolynomialFeatures(degree=2)
    x_ = pf.fit_transform(x)

    regr = LinearRegression()
    regr.fit(x_, y)
    y_ = regr.predict(x_)

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
    ax.plot(x, y_, color=colors[indexColor]["plotColor"], label="Modelo")
    ax.legend()

    plt.title(title)
    plt.xlabel("Días")
    plt.ylabel(y_label)
    plt.savefig("prediction.jpg")

    rmse = np.sqrt(mean_squared_error(y, y_))
    r2 = r2_score(y, y_)
    coef = regr.coef_
    intercept = regr.intercept_
    equation = f"y =  {coef[-1]} x^2 + {coef[-2]} x + {intercept}"

    # predict = (
    #     coef[-1] * (int(daysPredicted) ** 2) + coef[-2] * int(daysPredicted) + intercept
    # )

    x_min = int(daysPredicted)
    x_max = int(daysPredicted)
    x_new = np.linspace(x_min, x_max)
    x_new = x_new[:, np.newaxis]
    x_ = pf.fit_transform(x_new)

    predict = regr.predict(x_)[-1]

    return [rmse, r2, equation, intercept, predict, [x, y, y_]]


def genGraph(data1, data2):
    # title, y_label, x, y, y_
    fig, ax = plt.subplots()

    ax.scatter(data1[0], data1[1], color=colors[-1]["scatColor"], label="Casos")
    ax.plot(data1[0], data1[2], color=colors[-1]["plotColor"], label="Modelo casos")

    ax.scatter(data2[0], data2[1], color=colors[0]["scatColor"], label="Muertes")
    ax.plot(data2[0], data2[2], color=colors[0]["plotColor"], label="Modelo muertes")

    ax.legend()

    plt.title("Predicción de casos y muertes")
    plt.xlabel("Días")
    plt.ylabel("Casos / Muertes")
    plt.savefig("prediction.jpg")
