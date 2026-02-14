from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Pet DiagnosisService"
    app_env: str = "development"
    app_version: str = "1.0.0"

    # Optional OpenAI key — keep it optional so the app can run without it
    openai_api_key: str | None = None
    # OpenRouter (hosted) key — optional
    openrouter_api_key: str | None = None
    # Model name to use with OpenRouter (optional). If not set, route will use a sensible default.
    openrouter_model: str | None = None
    # Local LLM (llama.cpp/ggml) options
    local_llm_enabled: bool = False
    local_llm_model_path: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
