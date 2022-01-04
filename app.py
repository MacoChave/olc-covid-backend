import os
from analysis import getAnalysis
from fileManagment import saveDataFile, uploadImage
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from percentage import getPercentage

from prediction import getPredict
from rate import getRate
from trend import getTrend

FILENAME = "datafile"
UPLOAD_FOLDER = "./"
ALLOWED_EXTENSIONS = {"csv", "json", "xlsx", "xls"}

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["GET", "POST"])
@cross_origin()
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            # flash('No file part')
            return "No file part"
        file = request.files["file"]
        # IF USER DOES NOT SELECT FILE, BROWSER ALSO
        # SUBMIT AN EMPTY PART WITHOUT FILENAME
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            ext = request.args.get("ext")
            filename = f"dataFile.{ext}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return "Uploaded file"


@app.route("/uploadChunk", methods=["POST"])
@cross_origin()
def uploadChunk():
    return jsonify({"isSuccess": True})


@app.route("/uploadComplete", methods=["GET"])
@cross_origin()
def uploadComplete():
    return jsonify({"isSuccess": True})


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # try:
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
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
    # except:
    #     return jsonify(
    #         {
    #             "RMSE": "",
    #             "r^2": "",
    #             "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
    #             "Intercepto": "",
    #             "Predecir": "",
    #             "Grafica": "",
    #         }
    #     )


@app.route("/trend", methods=["POST"])
@cross_origin()
def trend():
    # try:
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
    res = getTrend(field, filtering, ext, sep, title)
    # graph = uploadImage("tendencia.jpg")
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
    # except:
    #     return jsonify(
    #         {
    #             "RMSE": "",
    #             "r^2": "",
    #             "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
    #             "Intercepto": "",
    #             "Predecir": "",
    #             "Grafica": "",
    #         }
    #     )


@app.route("/percentage", methods=["POST"])
@cross_origin()
def percentage():
    # try:
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
    res = getPercentage(field, filtering, ext, sep, title)
    # graph = uploadImage("percentage.jpg")
    return jsonify(
        {
            "RMSE": res[0],
            "r^2": res[1],
            "Ecuacion": res[2],
            "Intercepto": res[3],
            "Coeficiente": res[4],
            "Grafica": "graph",
            "Porcentaje": res[5],
        }
    )
    # except:
    #     return jsonify(
    #         {
    #             "RMSE": "",
    #             "r^2": "",
    #             "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
    #             "Intercepto": "",
    #             "Predecir": "",
    #             "Grafica": "",
    #         }
    #     )


@app.route("/rate", methods=["POST"])
@cross_origin()
def rate():
    # try:
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
    res = getRate(field, filtering, ext, sep, title)
    # graph = uploadImage("rate.jpg")
    return jsonify(
        {
            "RMSE": res[0],
            "r^2": res[1],
            "Ecuacion": res[2],
            "Intercepto": res[3],
            "Coeficiente": res[4],
            "Grafica": "graph",
            "Tasa": f"{res[5]}",
        }
    )
    # except:
    #     return jsonify(
    #         {
    #             "RMSE": "",
    #             "r^2": "",
    #             "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
    #             "Intercepto": "",
    #             "Predecir": "",
    #             "Grafica": "",
    #         }
    #     )


@app.route("/analysis", methods=["POST"])
@cross_origin()
def analysis():
    # try:
    json_data = request.get_json()
    ext = json_data["ext"]
    field = json_data["field"]
    filtering = json_data["filter"]
    sep = json_data["sep"]
    title = json_data["title"]
    res = getAnalysis(field, filtering, ext, sep, title)
    graph = uploadImage("analysis.jpg")

    return jsonify({"Grafica": graph, "Array": res})
    # except:
    #     return jsonify(
    #         {
    #             "RMSE": "",
    #             "r^2": "",
    #             "Ecuacion": "Tuvimos problemas técnicos para procesar el análisis. ¿Los datos están correctos?",
    #             "Intercepto": "",
    #             "Predecir": "",
    #             "Grafica": "",
    #         }
    #     )


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="8080")
    app.run(debug=True)
