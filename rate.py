from app import FILENAME
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from colors import colors
from pdf import genPDF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

IMAGENAME = "./reporte/rate.jpg"


def getRate(fields, filtering, ext, sep, title) -> list:
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
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Continente":
                countryField = filt["value"]

        df = filterRows(df, continentColumn, countryField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df))

        df_x["Tasa"] = df_x[confirmColumn] - df_x[deathColumn]

        total = df_x[confirmColumn].sum()
        tasa = df_x[confirmColumn].sum() - df_x[deathColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de crecimiento de casos activos en relación al númeor de muertes en {countryField}",
            "Infectados",
        )

        genPDF(title, IMAGENAME)
        return [pre[0], pre[1], pre[2], pre[3], pre[4], f"{tasa} / {total}"]
    elif (
        title
        == "Tasa de crecimiento de casos en relación con nuevos casos diarios y tasa de muerte"
    ):
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df_x))

        df_x["Tasa"] = df_x[confirmColumn] - df_x[deathColumn] - df_x[recoveryColumn]

        total = df_x[confirmColumn].sum()
        tasa = df_x[confirmColumn].sum() - df_x[deathColumn].sum()
        tasa = tasa - df_x[recoveryColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de crecimiento de casos en relación de casos diarios y tasa de muerte",
            "Infectados",
        )

        genPDF(title, IMAGENAME)
        return [pre[0], pre[1], pre[2], pre[3], pre[4], f"{tasa} / {total}"]
    else:  # Tasa de mortalidad por coronavirus (COVID-19) en un país
        countryField = ""
        for filt in filtering:
            if filt["key"] == "Pais":
                countryField = filt["value"]

        df = filterRows(df, countryColumn, countryField)
        df_ready = cleanRows(df, dateColumn)
        df_x = df_ready[0]
        df_x["Days"] = np.arange(len(df_x))

        df_x["Tasa"] = df_x[totalColumn] - df_x[deathColumn]

        total = df_x[totalColumn].sum()
        tasa = df_x[totalColumn].sum() - df_x[deathColumn].sum()

        pre = predict(
            np.asarray(df_x["Days"]).reshape(-1, 1),
            df_x["Tasa"],
            f"Tasa de mortalidad en {countryField}",
            "Muertes",
        )

        genPDF(title, IMAGENAME)
        return [pre[0], pre[1], pre[2], pre[3], pre[4], f"{tasa} / {total}"]


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


def writeDescriptionReport():
    f = open("./reporte/des.txt", "w")
    f.write(
        """ Para completar el analisis para encontrar la tendencia solicitada, primero se debio conocer todos los casos con sus respectivos días para poder dar la tendencia de los datos para del caso solicitado a partir de los datos recabados por medio del archivo de carga masiva.\n"""
    )
    f.write(
        """Para llegar a un resultado satisfactorio, se procedió a seguir los siguientes pasos:\n"""
    )
    f.write(
        """\t1. Leer el archivo de entrada
\t2. Rellenar los campos nulos con el elemento neutro, osea, ceros, esto para no ocacionar problemas durante el proceso
\t3. Hacer coincidir las columnas del archivo de entrada
\t4. Configurar el campo fecha del archivo de entrada
\t5. Limpiar las filas por el país solicitado
\t6. Separar la fecha por día, mes y año\n"""
    )
    f.write(
        f"\t7. Con la tendencia se podra observar si los datos siguen un comportamiento alcista o bajista, es decir, si los casos van aumentando o bajando.\n"
    )
    f.write("\t8. Se realizo lo mismo para el total de ambas columnas\n")
    f.write(
        """Completado los pasos anteriores, se procedio a generar los graficos pertinentes para demostrar los resultados teoricos obtenidos despues del analisis\n"""
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


def writeConclusionReport(res):
    f = open("./reporte/conc.txt", "w")
    f.write(""" Terminado el analisis, se llegaron a las siguientes concluciones:\n""")
    if res[4] > 0:
        f.write(
            f"Nos damos cuenta que el coeficiete es {res[4]} observando que los datos siguen un comportamiento alcista con los datos de muestra tomados. Por lo que podemos asegurar que estan aumentando los casos en estudio"
        )
    else:
        f.write(
            f"Nos damos cuenta que el coeficiete es {res[4]} observando que los datos siguen un comportamiento bajista con los datos de muestra tomados. Por lo que podemos asegurar que estan disminuyendo los casos en estudio"
        )
    f.write(
        f"La ecuacion que describe el comportamiento del porcentaje solicitado es: {res[2]}. Por lo que se observa se puede estimar los casos para un día futuro con dicha ecuación. La ecuación fue obtenida a partir de una regresión de grado 2.\n"
    )
    f.close()
