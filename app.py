from fileManagment import saveDataFile, uploadImage
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from prediction import getPredict
from trend import getTrend

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

FILENAME = "datafile"


@app.route("/")
@cross_origin()
def home():
    return jsonify(
        {
            "api": "COVID OLC",
            "author": "Marco Chavez",
            "fecha": "Diciembre 2021",
        }
    )


@app.route("/upload", methods=["POST"])
@cross_origin()
def upload():
    return "Upload POST"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        json_data = request.get_json()
        ext = json_data["ext"]
        field = json_data["field"]
        fileb64 = json_data["file"]
        filtering = json_data["filter"]
        sep = json_data["sep"]
        title = json_data["title"]
        saveDataFile(fileb64, ext)
        res = getPredict(field, filtering, ext, sep, title)
        # graph = uploadImage("prediction.jpg")
        return jsonify(
            {
                "RMSE": res[0],
                "r^2": res[1],
                "Ecuacion": res[2],
                "Intercepto": res[3],
                "Predecir": res[4],
                "Grafica": "graph",
            }
        )
    except:
        return jsonify(
            {
                "RMSE": "",
                "r^2": "",
                "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
                "Intercepto": "",
                "Predecir": "",
                "Grafica": "",
            }
        )


@app.route("/trend", methods=["POST"])
@cross_origin()
def trend():
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    fileb64 = json_data["file"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
    saveDataFile(fileb64, ext)
    res = getTrend(field, filtering, ext, sep, title)
    # graph = uploadImage("prediction.jpg")
    return jsonify(
        {
            "RMSE": res[0],
            "r^2": res[1],
            "Ecuacion": res[2],
            "Intercepto": res[3],
            "Coeficiente": res[4],
            "Grafica": "graph",
        }
    )


@app.route("/percentage", methods=["POST"])
@cross_origin()
def percentage():
    return jsonify({"message": "Percentage POST"})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="8080", debug=True)
    app.run(debug=True)
