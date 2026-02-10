from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Pet DiagnosisService"
    app_env: str = "development"
    app_version: str = "1.0.0"

    # Optional OpenAI key — keep it optional so the app can run without it
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
