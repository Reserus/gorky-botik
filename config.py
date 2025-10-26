import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_FILE_PATH = "data/cultural_objects_mnn.xlsx"
