import base64
from google.cloud import storage

# client = storage.Client()
# bucket = client.get_bucket("resources-covid19-olc")


def saveDataFile(fileb64: str, ext: str):
    with open(f"dataFile.{ext}", "wb") as fh:
        fh.write(base64.urlsafe_b64decode(fileb64))


def openImageB64(filename):
    encoded_string = ""
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def uploadImage(filename):
    blob = bucket.blob(f"resource/{filename}")
    blob.upload_from_filename(filename)
    print("Archivo subido")
    blob.make_public()
    print(blob)
    return blob.public_url
