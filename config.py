# config.py
import os

class Config:
    SECRET_KEY = os.urandom(24)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

