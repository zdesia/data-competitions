import os

from dotenv import load_dotenv

load_dotenv()

API_PORT = "8000"  # os.environ.get("API_PORT")
API_HOST = "0.0.0.0"  # os.environ.get("API_HOST")
API_VERSION = "v1"
TOKEN = os.environ.get("TOKEN")

URL_DATA_FOR_ANALYSIS = ["categories", "shopes", "sales"]
URL_HOLIDAY = "holiday"
URL_FORECAST = "forecast/custom_response_post"
LIMIT_REQUEST = 10000
LIMIT_POST = 1000
