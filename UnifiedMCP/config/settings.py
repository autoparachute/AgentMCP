from pydantic import BaseModel


class AppSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    root_path: str = ""
    rank_prefix: str = "/rank"
    calculate_prefix: str = "/calculate"


settings = AppSettings()


