
import redis
import pickle
from vinagent.config.logger_config import setup_logger
from typing import Any

logger = setup_logger(__name__,"vinagent_analysis.log") 

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, default_ttl=3600):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.default_ttl = default_ttl
        logger.info(f"Connected to Redis at {host}:{port}, db={db}")

    def get(self, key: str) -> Any:
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Redis cache hit for key: {key}")
                return pickle.loads(value)
            logger.debug(f"Redis cache miss for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving key '{key}' from Redis: {e}", exc_info=True)
            return None
        
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        try:
            self.client.setex(key, ttl or self.default_ttl, pickle.dumps(value))
            logger.debug(f"Set Redis key: {key} with TTL={ttl or self.default_ttl}")
        except Exception as e:
            logger.error(f"Error setting key '{key}' in Redis: {e}", exc_info=True)

    def exists(self, key: str) -> bool:
        try:
            return self.client.exists(key) == 1
        except Exception as e:
            logger.error(f"Error checking existence of key '{key}': {e}", exc_info=True)
            return False