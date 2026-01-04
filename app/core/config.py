from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Pet DiagnosisService"
    app_env: str = "development"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"


settings = Settings()
