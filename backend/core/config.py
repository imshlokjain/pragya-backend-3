from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+psycopg2://pragya:pragya@localhost:5432/pragya"
    )
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "development"
    pilot_district_id: str = "D001"
    model_provider: str = "prototype"

    jwt_secret: str = "CHANGE_THIS_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://localhost:5174"

    model_config = SettingsConfigDict(
        env_file=".env",
        protected_namespaces=("settings_",),
    )


settings = Settings()
