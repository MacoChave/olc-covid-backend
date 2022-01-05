import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from colors import colors
from pdf import PDF, genPDF

IMAGENAME = "./reporte/prediction.jpg"


def getPredict(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    df.fillna(0)

    countryColumn = ""
    deptoColumn = ""
    dateColumn = ""
    confirmColumn = ""
    deathColumn = ""

    # GET COLUMN NAMES
    for item in fields:
        if item["require"] == "Pais":
            countryColumn = item["match"]
        if item["require"] == "Departamento":
            deptoColumn = item["match"]
        if item["require"] == "Fecha":
            dateColumn = item["match"]
        if item["require"] == "Confirmados":
            confirmColumn = item["match"]
        if item["require"] == "Muertes":
            deathColumn = item["match"]

    # CREATE YEAR, MONTH COLUMNS FROM DATE COLUMN
    # dateRow = df.iloc[0][dateColumn]
    df["JoinedDate"] = pd.to_datetime(df[dateColumn], infer_datetime_format=True)
    df["Year"] = df["JoinedDate"].dt.year
    df["Month"] = df["JoinedDate"].dt.month

    if title == "Prediccion de infectados en un Pais":
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
        print(df_x)
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysField,
            f"Prediccion de infectados en {countryField}",
            "Infectados",
        )
        pasos = f" 7. Predecir los infectados para {daysField} dias\n"
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, daysField)

        genPDF(title, IMAGENAME)
        return pre
    elif title == "Prediccion de mortalidad por COVID en un Departamento":
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
            f"Prediccion de mortalidad en {deptoField}, {countryField}",
            "Muertes",
        )
        pasos = f""" 7. Prediccion de la moralidad para {daysField} dias"""
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, daysField)

        genPDF(title, IMAGENAME)
        return pre
    elif title == "Prediccion de mortalidad por COVID en un Pais":
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
            f"Prediccion de mortalidad en {countryField}",
            "Muertes",
        )
        pasos = """ 7. """
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, daysField)

        genPDF(title, IMAGENAME)
        return pre
    elif title == "Prediccion de casos de un pais para un año":
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
            f"Prediccion de casos en {countryField} para {yearField} años",
            "Confirmados",
        )
        pasos = f""" 7. Prediccion de la moralidad para {daysPredicted} dias"""
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, daysPredicted)

        genPDF(title, IMAGENAME)
        return pre
    elif (
        title
        == "Prediccion de muertes en el último dia del primer año de infecciones en un pais"
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
        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            365,
            f"Prediccion de mortalidad en {countryField} para el último dia del primer año",
            "Muertes",
        )
        pasos = f""" 7. Prediccion de la moralidad para el 1er año de infecciones"""
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, 365)

        genPDF(title, IMAGENAME)
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
        if len(df_x) > len(df_y):
            df_x = df_x[:-1]
        else:
            df_y = df_y[:-1]
        print(df_x)
        print(df_y)
        pre_confirms = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[confirmColumn],
            daysField,
            f"Prediccion de casos",
            "Confirmados",
        )
        pre_deaths = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_y[deathColumn],
            daysField,
            f"Prediccion de mortalidad",
            "Muertes",
        )
        genGraph(pre_confirms[5], pre_deaths[5])
        return [
            pre_confirms[0],
            pre_confirms[1],
            f"Casos = {pre_confirms[2]} - Muertes = {pre_deaths[2]}",
            pre_confirms[3],
            f"Casos = {pre_confirms[4]} - Muertes = {pre_deaths[4]}",
        ]
    else:  # Prediccion de casos confirmados por dia
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
            f"Prediccion de casos confirmados por dia",
            "Confirmados",
        )
        pasos = f""" 7. Prediccion de la moralidad para {daysField} dias"""
        writeDescriptionReport(pasos)
        writeDatosReport(pre)
        writeConclusionReport(pre, daysField)

        genPDF(title, IMAGENAME)
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
    plt.xlabel("Dias")
    plt.ylabel(y_label)
    plt.savefig(FILENAME)

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

    return [rmse, r2, equation, intercept, predict, [x, y, y_], coef]


def genGraph(data1, data2):
    # title, y_label, x, y, y_
    fig, ax = plt.subplots()

    ax.scatter(data1[0], data1[1], color=colors[-1]["scatColor"], label="Casos")
    ax.plot(data1[0], data1[2], color=colors[-1]["plotColor"], label="Modelo casos")

    ax.scatter(data2[0], data2[1], color=colors[0]["scatColor"], label="Muertes")
    ax.plot(data2[0], data2[2], color=colors[0]["plotColor"], label="Modelo muertes")

    ax.legend()

    plt.title("Prediccion de casos y muertes")
    plt.xlabel("Dias")
    plt.ylabel("Casos / Muertes")
    plt.savefig(IMAGENAME)


def writeDescriptionReport(pasos):
    f = open("./reporte/des.txt", "w")
    f.write(
        """ Para completar el analisis de prediccion solicitado, primero se debio conocer un dia futuro para predecir el caso solicitado en ese punto a partir de los datos recabados por medio del archivo de carga masiva.\n"""
    )
    f.write(
        """Para llegar a un resultado satisfactorio, se procedio a seguir los siguientes pasos:\n"""
    )
    f.write(
        """ 1. Leer el archivo de entrada
    2. Rellenar los campos nulos con el elemento neutro, osea, ceros, esto para no ocacionar problemas durante el proceso
    3. Hacer coincidir las columnas del archivo de entrada
    4. Configurar el campo fecha del archivo de entrada
    5. Limpiar las filas por el pais solicitado
    6. Separar la fecha por dia, mes y año"""
    )
    f.writelines(pasos)
    f.write(
        """Completado los pasos anteriores, se procedio a generar los graficos pertinentes para demostrar los resultados teoricos obtenidos despues del analisis"""
    )
    f.close()


def writeDatosReport(res):
    f = open("./reporte/datos.txt", "w")
    f.write(""" Los datos recabados con el analisis son los siguientes:\n""")
    f.write(f"  - El indice RMSE obtenido es: {res[0]}\n")
    f.write(f"  - El r^2 obtenido es: {res[1]}\n")
    f.write(f"  - El punto de corte es: {res[3]}\n")
    f.write(f"  - El coeficiente es: {res[6]}\n")
    f.write(f"  - La ecuacion obtenida es: {res[2]}\n")
    f.close()


def writeConclusionReport(res, time):
    f = open("./reporte/conc.txt", "w")
    f.write(""" Terminado el analisis, se llegaron a las siguientes concluciones:\n""")
    f.write(
        f"La ecuacion que describe el comportamiento para {time} dias es : {res[4]}. Por lo que se observa se puede estimar los casos para un dia futuro con dicha ecuacion. La ecuacion fue obtenida a partir de una regresion de grado 2.\n"
    )
    f.close()
