from pandas.core.frame import DataFrame
from fileManagment import openImageB64
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def getPredict(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    countryColumn = ""
    deptoColumn = ""
    regionColumn = ""
    dateColumn = ""
    confirmColumn = ""
    deathColumn = ""
    recoveredColumn = ""
    genreColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Region":
            regionColumn = item["match"]
        if item["require"] == "Departamento":
            deptoColumn = item["match"]
        if item["require"] == "Region":
            regionColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Confirmados":
            confirmColumn = item["match"]
        if item["require"] == "Muertes":
            deathColumn = item["match"]
        if item["require"] == "Recuperados":
            recoveredColumn = item["match"]
        if item["require"] == "Genero":
            genreColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    df["JoinedDate"] = pd.to_datetime(df[dateColumn])
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if title == "Predicción de infectados en un País":
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
            df_y[confirmColumn],
            daysField,
            f"Predicción de infectados en {countryField}",
            "Infectados",
        )
        return pre
    elif title == "Predicción de mortalidad por COVID en un Departamento":
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
            f"Predicción de mortalidad en {deptoField}, {countryField}",
            "Muertes",
        )
        return pre
    elif title == "Predicción de mortalidad por COVID en un País":
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
            f"Predicción de mortalidad en {countryField}",
            "Muertes",
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
    elif (
        title
        == "Predicción de muertes en el último día del primer año de infecciones en un país"
    ):
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]

        df = filterRows(df, countryColumn, countryColumn)
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        # TODO: predictToEndYear
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            365,
            f"Predicción de mortalidad en {countryField} para el último día del primer año",
            "Muertes",
        )
        return pre
    elif (
        title
        == "Predicciones de casos y muertes en todo el mundo - Neural Network MLPRegressor"
    ):
        daysField = ""
        for filt in filtering:
            if filt["key"] == "Dias":

                daysField = filt["value"]
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        pre_confirms = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysField,
            f"Predicción de mortalidad",
            "Confirmados",
        )
        pre_deaths = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            daysField,
            f"Predicción de mortalidad",
            "Muertes",
        )
        return pre_confirms
    else:  # Predicción de casos confirmados por día
        daysField = ""
        for filt in filtering:
            if filt["key"] == "Dias":
                daysField = filt["value"]
        df["Days"] = np.arange(len(df))
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_y = df_ready[1]
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysField,
            f"Predicción de casos confirmados por día",
            "Confirmados",
        )
        return pre


def filterRows(dataFrame: DataFrame, columnName: str, rowName: str) -> DataFrame:
    print(f"filtrar {columnName} por {rowName}")
    dataFrame = dataFrame.loc[lambda x: x[columnName] == rowName]
    return dataFrame


def cleanRows(dataFrame: DataFrame, date):
    df_cleaned = dataFrame.drop_duplicates(date)
    df_summed = dataFrame.groupby(by=["Year", "Month", date]).sum()
    return [df_cleaned, df_summed]


def predict(x, y, daysPredicted: int, title: str, y_label: str) -> list:
    pf = PolynomialFeatures(degree=2)
    x_ = pf.fit_transform(x)

    regr = LinearRegression()
    regr.fit(x_, y)
    y_ = regr.predict(x_)

    plt.title(title)
    plt.ylabel("Días")
    plt.xlabel(y_label)
    plt.scatter(x, y, color="gray")
    plt.plot(x, y_, color="red")

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

    return [rmse, r2, equation, intercept, predict]
