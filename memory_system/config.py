"""Configuration management for the AI Memory System."""

import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))

    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

class MongoDBConfig(BaseModel):
    """MongoDB configuration."""
    host: str = Field(default_factory=lambda: os.getenv("MONGO_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("MONGO_PORT", "27017")))
    username: Optional[str] = Field(default_factory=lambda: os.getenv("MONGO_USERNAME"))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("MONGO_PASSWORD"))
    database: str = Field(default_factory=lambda: os.getenv("MONGO_DATABASE", "memory_system"))
    
    @property
    def uri(self) -> str:
        """Get MongoDB URI."""
        auth = f"{self.username}:{self.password}@" if self.username and self.password else ""
        return f"mongodb://{auth}{self.host}:{self.port}/{self.database}"

class KafkaConfig(BaseModel):
    """Kafka configuration."""
    bootstrap_servers: str = Field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    memory_topic: str = Field(
        default_factory=lambda: os.getenv("KAFKA_MEMORY_TOPIC", "memory-events")
    )

class RayConfig(BaseModel):
    """Ray configuration."""
    address: Optional[str] = Field(default_factory=lambda: os.getenv("RAY_ADDRESS"))
    num_replicas: int = Field(default_factory=lambda: int(os.getenv("RAY_NUM_REPLICAS", "2")))

class ModelConfig(BaseModel):
    """Model configuration."""
    model_id: str = Field(default_factory=lambda: os.getenv("MODEL_ID", "gpt2"))
    max_input_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_INPUT_TOKENS", "4096")))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))

class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

class Config(BaseModel):
    """Main configuration."""
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a global config instance
config = Config() 
