import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from pdf import genPDF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

IMAGENAME = "./reporte/percentage.jpg"


def getPercentage(fields, filtering, ext, sep, title) -> list:
    df = None

    if ext == "csv":
        df = pd.read_csv("dataFile.csv", sep)
    elif ext == "json":
        df = pd.read_json("dataFile.json")
    else:
        df = pd.read_excel("dataFile.xlsx")

    df.fillna(0)

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

        df = df.loc[df[confirmColumn] != 0, :]

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

        writeDescriptionReport(genreColumn, confirmColumn)
        writeDatosReport(pre)
        writeConclusionReport(pre, percent)

        genPDF(title, IMAGENAME)
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

        writeDescriptionReport(genreColumn, confirmColumn)
        writeDatosReport(pre)
        writeConclusionReport(pre, percent)

        genPDF(title, IMAGENAME)
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
    plt.savefig(FILENAME)

    rmse = np.sqrt(mean_squared_error(y, y_))
    r2 = r2_score(y, y_)
    coef = regr.coef_
    intercept = regr.intercept_
    equation = f"y =  {coef[-1]} b + {intercept}"

    return [rmse, r2, equation, intercept, coef[-1]]


def writeDescriptionReport(individual, total):
    f = open("./reporte/des.txt", "w")
    f.write(
        """ Para completar el analisis para encontrar el porcentaje solicitado, primero se debio conocer el total de casos para poder dar el porcentaje del caso solicitado para cada fila a partir de los datos recabados por medio del archivo de carga masiva.\n"""
    )
    f.write(
        """Para llegar a un resultado satisfactorio, se procedió a seguir los siguientes pasos:\n"""
    )
    f.write(
        """ 1. Leer el archivo de entrada
    2. Rellenar los campos nulos con el elemento neutro, osea, ceros, esto para no ocacionar problemas durante el proceso
    3. Hacer coincidir las columnas del archivo de entrada
    4. Configurar el campo fecha del archivo de entrada
    5. Limpiar las filas por el país solicitado
    6. Separar la fecha por día, mes y año"""
    )
    f.write(
        f"  7. El porcentaje para cada fila, se obtuvo de la siguiente operación: {individual} * 100 / {total}"
    )
    f.write("   8. Se realizo lo mismo para el total de ambas columnas")
    f.write(
        """Completado los pasos anteriores, se procedio a generar los gráficos pertinentes para demostrar los resultados teoricos obtenidos despues del analisis"""
    )
    f.close()


def writeDatosReport(res):
    f = open("./reporte/datos.txt", "w")
    f.write(""" Los datos recabados con el analisis son los siguientes:\n""")
    f.write(f"  - El indice RMSE obtenido es: {res[0]}\n")
    f.write(f"  - El r^2 obtenido es: {res[1]}\n")
    f.write(f"  - El punto de corte es: {res[3]}\n")
    f.write(f"  - El coeficiente es: {res[4]}\n")
    f.write(f"  - La ecuacion obtenida es: {res[2]}\n")
    f.close()


def writeConclusionReport(res, percent):
    f = open("./reporte/conc.txt", "w")
    f.write(""" Terminado el análisis, se llegaron a las siguientes concluciones:\n""")
    f.write(
        f"La ecuacion que describe el comportamiento del porcentaje solicitado es: {res[2]}. Por lo que se observa se puede estimar los casos para un día futuro con dicha ecuación. La ecuación fue obtenida a partir de una regresión de grado 2.\n"
    )
    f.write(
        f"Pero, tambien obtenemos el porcentaje total obtenido, siendo de {percent}"
    )
    f.close()
