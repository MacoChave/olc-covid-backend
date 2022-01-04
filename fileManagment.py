import json
import base64
import boto3
from botocore.exceptions import NoCredentialsError

with open("credentials.json", "r") as creds:
    data = creds.read()

ACCESS_KEY = json.loads(data)["ACCESS_KEY"]
SECRET_KEY = json.loads(data)["SECRET_KEY"]
BUCKET = "res-covid19-olc"


def saveDataFile(fileb64: str, ext: str):
    with open(f"fileData.{ext}", "wb") as fh:
        fh.write(base64.urlsafe_b64decode(fileb64))


def openImageB64(filename):
    encoded_string = ""
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def uploadImage(filename):
    s3 = boto3.client(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )
    with open(filename, "rb") as file:
        try:
            s3.upload_fileobj(file, BUCKET, filename, ExtraArgs={"ACL": "public-read"})
            print("Upload Successful")
            return f"https://res-covid19-olc.s3.us-east-2.amazonaws.com/{filename}"
        except FileNotFoundError:
            print("The file was not found")
            return f"https://res-covid19-olc.s3.us-east-2.amazonaws.com/{filename}"
        except NoCredentialsError:
            print("Credentials not available")
            return f"https://res-covid19-olc.s3.us-east-2.amazonaws.com/{filename}"
