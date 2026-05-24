from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: SecretStr

    model_config = SettingsConfigDict(env_file=".env.local")
    
settings = Settings()

print(settings.groq_api_key)


