import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_SERVER = os.getenv(
    "DB_SERVER",
    r"DESKTOP-5EHP310\SQLEXPRESS",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "EnterpriseRAG",
)

DB_DRIVER = os.getenv(
    "DB_DRIVER",
    "ODBC Driver 18 for SQL Server",
)

DATABASE_URL = (
    "mssql+pyodbc://@"
    + DB_SERVER
    + "/"
    + DB_NAME
    + "?driver="
    + DB_DRIVER.replace(" ", "+")
    + "&trusted_connection=yes"
    + "&TrustServerCertificate=yes"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)