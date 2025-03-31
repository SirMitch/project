# File: serenity/memory/manager.py
# Created: 2024-12-24 15:40:46
# Purpose: Memory Management System
# Version: 1.0

"""
Serenity Memory Management System
Handles data storage, retrieval, and embeddings for AI context
"""

import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import threading
import json

class MemoryManager:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Memory')
        self.memory_path = Path('serenity/data/memory')
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.config = None
        self.events = None
        self.db_conn = None
        self.cache = {}  # Simple in-memory cache
        self.running = False
        self.lock = threading.Lock()

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.logger.info("Memory configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("response_generated", self.store_interaction)
        self.events.subscribe("system_shutdown", self.stop)
        self.logger.info("Memory event handler set")

    def initialize(self):
        """Initialize memory system with database."""
        try:
            if not self.config:
                raise ValueError("Configuration not set")

            self.logger.info("Initializing memory manager...")
            db_path = self.memory_path / 'memory.db'
            self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._setup_database()
            self.cache = {}
            self.logger.info("Memory manager initialized")
        except Exception as e:
            self.logger.error(f"Memory initialization failed: {str(e)}")
            raise

    def start(self):
        """Start the memory manager."""
        try:
            if not self.events:
                raise ValueError("Event handler not set")
            self.running = True
            self.initialize()
            self.logger.info("Memory manager started")
        except Exception as e:
            self.logger.error(f"Memory startup failed: {str(e)}")
            self.stop()

    def _setup_database(self):
        """Set up SQLite tables for memory storage."""
        try:
            with self.lock:
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        input_text TEXT,
                        response_text TEXT,
                        embedding BLOB
                    )
                ''')
                self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Database setup failed: {str(e)}")
            raise

    def store_interaction(self, event_data: Dict):
        """Store an interaction from AIProcessor."""
        try:
            if not self.running:
                self.logger.warning("Memory manager not running")
                return

            input_text = event_data.get('input', '')
            response = event_data.get('response', '')
            timestamp = event_data.get('timestamp', '')

            if not input_text or not response:
                self.logger.debug("Invalid interaction data")
                return

            embedding = self._generate_embedding(input_text + " " + response)
            with self.lock:
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "INSERT INTO interactions (timestamp, input_text, response_text, embedding) VALUES (?, ?, ?, ?)",
                    (timestamp, input_text, response, embedding.tobytes())
                )
                self.db_conn.commit()

                # Cache latest interaction
                interaction_id = cursor.lastrowid
                self.cache[interaction_id] = {
                    "timestamp": timestamp,
                    "input": input_text,
                    "response": response,
                    "embedding": embedding
                }
                if len(self.cache) > self.config.get('memory', 'cache_size', 1024):
                    self.cache.pop(next(iter(self.cache)))

                self.logger.debug(f"Stored interaction ID {interaction_id}")
                self.events.emit("memory_stored", {"id": interaction_id}, priority=4)

        except Exception as e:
            self.logger.error(f"Interaction storage failed: {str(e)}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding for text."""
        try:
            # Fake embedding for now—real one would use a model like SentenceTransformers
            hash_val = hash(text) % 1000
            embedding = np.array([hash_val + i * 0.1 for i in range(128)], dtype=np.float32)
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return np.zeros(128, dtype=np.float32)

    def retrieve_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant past interactions based on query."""
        try:
            query_embedding = self._generate_embedding(query)
            with self.lock:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT id, timestamp, input_text, response_text, embedding FROM interactions ORDER BY id DESC LIMIT 100")
                rows = cursor.fetchall()

            # Simple cosine similarity for relevance
            results = []
            for row in rows:
                id, timestamp, input_text, response_text, embedding_blob = row
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                results.append({
                    "id": id,
                    "timestamp": timestamp,
                    "input": input_text,
                    "response": response_text,
                    "similarity": float(similarity)
                })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Context retrieval failed: {str(e)}")
            return []

    def get_memory_stats(self) -> Dict:
        """Get memory system statistics."""
        try:
            with self.lock:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM interactions")
                total = cursor.fetchone()[0]
            return {
                "total_interactions": total,
                "cache_size": len(self.cache),
                "timestamp": self._get_current_time()
            }
        except Exception as e:
            self.logger.error(f"Memory stats failed: {str(e)}")
            return {}

    def _get_current_time(self) -> str:
        """Helper to get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def stop(self, event_data=None):
        """Stop the memory manager and clean up."""
        try:
            self.running = False
            if self.db_conn:
                self.db_conn.close()
            self.cache.clear()
            self.logger.info("Memory manager stopped")
        except Exception as e:
            self.logger.error(f"Memory shutdown failed: {str(e)}")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    try:
        memory = MemoryManager()
        config = ConfigManager()
        events = EventHandler()
        memory.set_config(config)
        memory.set_events(events)
        memory.start()

        events.emit("response_generated", {
            "input": "Hello, how are you?",
            "response": "I'm good, thanks!",
            "timestamp": memory._get_current_time()
        })
        import time
        time.sleep(1)  # Let it store
        context = memory.retrieve_context("Hello")
        print("Retrieved context:", json.dumps(context, indent=2))
        memory.stop()

    except Exception as e:
        logging.getLogger('Serenity.Memory').error(f"Test failed: {str(e)}")