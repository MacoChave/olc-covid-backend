from fileManagment import saveDataFile, uploadImage
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from prediction import getPredict

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
    # print('Imagen en: ', graph)
    graph = ""
    return jsonify(
        {
            "RMSE": res[0],
            "r^2": res[1],
            "Ecuacion": res[2],
            "Intercepto": res[3],
            "Predecir": res[4],
            "Grafica": graph,
        }
    )
    # try:
    # except:
    #     return jsonify({"Message": "Error en analizar entrada"})


@app.route("/trend", methods=["POST"])
@cross_origin()
def trend():
    return jsonify({"message": "Tendence POST"})


@app.route("/percentage", methods=["POST"])
@cross_origin()
def percentage():
    return jsonify({"message": "Percentage POST"})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="8080", debug=True)
    app.run(debug=True)
