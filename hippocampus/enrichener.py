#!/usr/bin/env python3
"""
Enrichener - Kafka consumer that retrieves context for prompts.

Consumes from 'ai_in' topic, uses LLM to decide if context retrieval is needed,
queries ChromaDB for relevant context, stores context in context.db,
and forwards prompts to 'prompts' topic.

Data Flow:
    User Input -> [ai_in topic] -> Enrichener -> context.db (stores context)
                                              -> [prompts topic] -> Cortex
"""

from kafka import KafkaConsumer, KafkaProducer
import json
import os
import sys
import time
from datetime import datetime
from openai import OpenAI
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import chromadb
from chromadb.config import Settings

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.embeddings import Embedder
from libs.olorin_logging import OlorinLogger
from libs.context_store import ContextStore

# Initialize config with hot-reload support
config = Config(watch=True)

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "hippocampus-enrichener.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

# Initialize logger
logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


# =============================================================================
# LLM PROMPT TEMPLATES
# =============================================================================

DECISION_SYSTEM_PROMPT = """You are a query classifier. Your job is to determine if a user query would benefit from additional context from a knowledge base.

Respond with ONLY "YES" or "NO".

Answer "YES" if the query:
- Asks about specific topics, facts, or information that could be in documents
- Requests explanations, summaries, or details about concepts
- References proper nouns, technical terms, or domain-specific knowledge
- Could be answered more accurately with reference material

Answer "NO" if the query:
- Is a simple greeting or casual conversation
- Asks for creative writing with no factual basis needed
- Is a meta-question about the AI itself
- Is a simple command or instruction that doesn't need context
- Is already self-contained with all necessary information"""

DECISION_USER_TEMPLATE = """Query: {prompt}

Does this query need context enrichment? Answer YES or NO only."""


# =============================================================================
# CONFIGURATION
# =============================================================================


class EnrichenerConfig:
    """Configuration wrapper for Enrichener consumer"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        """Load configuration values from Config"""
        # Kafka settings
        self.kafka_bootstrap_servers = self.cfg.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.kafka_input_topic = self.cfg.get("ENRICHENER_INPUT_TOPIC", "ai_in")
        self.kafka_output_topic = self.cfg.get("ENRICHENER_OUTPUT_TOPIC", "prompts")
        self.kafka_broca_topic = self.cfg.get("BROCA_KAFKA_TOPIC", "ai_out")
        self.kafka_consumer_group = self.cfg.get(
            "ENRICHENER_CONSUMER_GROUP", "enrichener-consumer-group"
        )
        self.kafka_auto_offset_reset = self.cfg.get(
            "ENRICHENER_AUTO_OFFSET_RESET", "earliest"
        )

        # Exo/LLM settings
        self.exo_base_url = self.cfg.get("EXO_BASE_URL", "http://localhost:52415/v1")
        self.exo_api_key = self.cfg.get("EXO_API_KEY", "dummy-key")

        # MODEL_NAME is optional - if empty/not set, will auto-detect from running instance
        model_name_env = self.cfg.get("MODEL_NAME", "").strip()
        self.model_name = model_name_env if model_name_env else None

        self.llm_timeout_seconds = self.cfg.get_int("LLM_TIMEOUT_SECONDS", 120)
        self.decision_temperature = self.cfg.get_float("DECISION_TEMPERATURE", 0.1)

        # ChromaDB settings
        self.chromadb_host = self.cfg.get("CHROMADB_HOST", "localhost")
        self.chromadb_port = self.cfg.get_int("CHROMADB_PORT", 8000)
        self.chromadb_collection = self.cfg.get("CHROMADB_COLLECTION", "documents")
        self.chromadb_n_results = self.cfg.get_int("CHROMADB_QUERY_N_RESULTS", 10)

        # Context store settings
        # Path resolved via config library against project root
        self.context_db_path = self.cfg.get_path(
            "CONTEXT_DB_PATH", "./hippocampus/data/context.db"
        )

        # Thread pool
        self.thread_pool_size = self.cfg.get_int("ENRICHENER_THREAD_POOL_SIZE", 3)

        # Logging
        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

    def reload(self) -> bool:
        """Check for config changes and reload if needed"""
        if self.cfg.reload():
            self._load()
            return True
        return False


def load_config() -> EnrichenerConfig:
    """Load configuration from environment variables"""
    logger.info("Loading configuration from environment variables...")

    enrichener_cfg = EnrichenerConfig(config)

    # Update logging level
    import logging

    logger.setLevel(getattr(logging, enrichener_cfg.log_level.upper(), logging.INFO))

    logger.info("Configuration loaded successfully:")
    logger.info(f"  Kafka Bootstrap Servers: {enrichener_cfg.kafka_bootstrap_servers}")
    logger.info(f"  Kafka Input Topic: {enrichener_cfg.kafka_input_topic}")
    logger.info(f"  Kafka Output Topic: {enrichener_cfg.kafka_output_topic}")
    logger.info(f"  Kafka Broca Topic: {enrichener_cfg.kafka_broca_topic}")
    logger.info(f"  Kafka Consumer Group: {enrichener_cfg.kafka_consumer_group}")
    logger.info(f"  Exo Base URL: {enrichener_cfg.exo_base_url}")
    logger.info(
        f"  Model Name: {enrichener_cfg.model_name if enrichener_cfg.model_name else 'auto-detect'}"
    )
    logger.info(f"  LLM Timeout: {enrichener_cfg.llm_timeout_seconds}s")
    logger.info(f"  Decision Temperature: {enrichener_cfg.decision_temperature}")
    logger.info(
        f"  ChromaDB: {enrichener_cfg.chromadb_host}:{enrichener_cfg.chromadb_port}"
    )
    logger.info(f"  ChromaDB Collection: {enrichener_cfg.chromadb_collection}")
    logger.info(f"  ChromaDB Results: {enrichener_cfg.chromadb_n_results}")
    logger.info(f"  Context DB: {enrichener_cfg.context_db_path}")
    logger.info(f"  Thread Pool Size: {enrichener_cfg.thread_pool_size}")
    logger.info(f"  Log Level: {enrichener_cfg.log_level}")

    return enrichener_cfg


# =============================================================================
# MAIN ENRICHER CLASS
# =============================================================================


class PromptEnricher:
    """
    Main consumer class that retrieves context for prompts.

    Processing Pipeline:
    1. Consume prompt from ai_in topic
    2. Decide if prompt needs context retrieval (LLM call)
    3. Query ChromaDB for relevant context
    4. Store context chunks in context.db with metadata
    5. Forward original prompt to prompts topic
    """

    def __init__(self, enrichener_config: EnrichenerConfig):
        logger.info("Initializing PromptEnricher...")
        self.config = enrichener_config

        # Initialize Kafka consumer
        self._init_kafka_consumer()

        # Initialize Kafka producer
        self._init_kafka_producer()

        # Initialize ChromaDB client and collection
        self._init_chromadb()

        # Initialize context store
        self._init_context_store()

        # Initialize embedding model
        self._init_embedding_model()

        # Initialize OpenAI client pointing to Exo
        self.client = self._init_openai_client()

        # Initialize thread pool for message processing
        self.executor = ThreadPoolExecutor(
            max_workers=config.thread_pool_size, thread_name_prefix="enricher-worker"
        )
        self.active_futures = []
        self.shutdown_event = threading.Event()
        logger.info(
            f"Thread pool executor initialized with {config.thread_pool_size} workers"
        )

        logger.info("PromptEnricher initialized successfully")
        logger.info(f"  Input topic: {config.kafka_input_topic}")
        logger.info(f"  Output topic: {config.kafka_output_topic}")
        logger.info(f"  Context DB: {config.context_db_path}")

    def _init_kafka_consumer(self):
        """Initialize Kafka consumer with anti-rebalance settings"""
        logger.info(
            f"Creating Kafka consumer for topic '{self.config.kafka_input_topic}'..."
        )
        logger.info(f"  Bootstrap servers: {self.config.kafka_bootstrap_servers}")
        logger.info(f"  Consumer group: {self.config.kafka_consumer_group}")
        try:
            self.consumer = KafkaConsumer(
                self.config.kafka_input_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_deserializer=lambda m: m.decode("utf-8"),
                auto_offset_reset=self.config.kafka_auto_offset_reset,
                enable_auto_commit=True,
                group_id=self.config.kafka_consumer_group,
                # Critical: prevent rebalance during long LLM calls
                max_poll_interval_ms=600000,  # 10 minutes
                session_timeout_ms=60000,  # 60 seconds
                heartbeat_interval_ms=10000,  # 10 seconds
            )
            logger.info("Kafka consumer created successfully")
            logger.info("  max_poll_interval_ms: 600000 (10 minutes)")
            logger.info("  session_timeout_ms: 60000 (60 seconds)")
            logger.info("  heartbeat_interval_ms: 10000 (10 seconds)")
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}", exc_info=True)
            raise

    def _init_kafka_producer(self):
        """Initialize Kafka producer for output"""
        logger.info(
            f"Creating Kafka producer for output topic '{self.config.kafka_output_topic}'..."
        )
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka producer created successfully")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}", exc_info=True)
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB client with connection retry"""
        logger.info(
            f"Connecting to ChromaDB at {self.config.chromadb_host}:{self.config.chromadb_port}..."
        )
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.chromadb_client = chromadb.HttpClient(
                    host=self.config.chromadb_host,
                    port=self.config.chromadb_port,
                    settings=Settings(anonymized_telemetry=False),
                )
                self.collection = self.chromadb_client.get_collection(
                    name=self.config.chromadb_collection
                )
                doc_count = self.collection.count()
                logger.info("Connected to ChromaDB successfully")
                logger.info(
                    f"  Collection '{self.config.chromadb_collection}' has {doc_count} documents"
                )
                return
            except Exception as e:
                logger.warning(
                    f"ChromaDB connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error("Failed to connect to ChromaDB after all retries")
                    # Set to None - will cause graceful pass-through on queries
                    self.chromadb_client = None
                    self.collection = None

    def _init_context_store(self):
        """Initialize context store for storing retrieved contexts"""
        logger.info(f"Initializing context store at {self.config.context_db_path}...")
        try:
            self.context_store = ContextStore(self.config.context_db_path)
            stats = self.context_store.get_statistics()
            logger.info("Context store initialized successfully")
            logger.info(f"  Existing contexts: {stats['total_contexts']}")
            logger.info(f"  Unique prompts: {stats['unique_prompts']}")
        except Exception as e:
            logger.error(f"Failed to initialize context store: {e}", exc_info=True)
            raise

    def _init_embedding_model(self):
        """Initialize embedder (shared singleton)"""
        try:
            self.embedder = Embedder.get_instance()
            logger.info(
                f"Embedder ready: {self.embedder.model_name} (dimension: {self.embedder.dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}", exc_info=True)
            raise

    def _init_openai_client(self):
        """Initialize OpenAI client pointing to Exo"""
        logger.info(
            f"Initializing OpenAI client for Exo at {self.config.exo_base_url}..."
        )
        client = OpenAI(
            base_url=self.config.exo_base_url, api_key=self.config.exo_api_key
        )
        logger.info("OpenAI client initialized successfully")
        return client

    def _get_running_model(self) -> str | None:
        """Query Exo to get the currently running model"""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Auto-detecting running model from Exo...")

        try:
            base_url = self.config.exo_base_url.rstrip("/v1").rstrip("/")
            state_url = f"{base_url}/state"
            logger.debug(f"[{thread_name}] Querying: {state_url}")

            response = requests.get(state_url, timeout=5)
            response.raise_for_status()
            state = response.json()

            instances = state.get("instances", {})
            runners = state.get("runners", {})
            logger.debug(f"[{thread_name}] Found {len(instances)} instance(s)")

            if not instances:
                logger.warning(f"[{thread_name}] No instances in state")
                return None

            # Check each instance for a ready model
            for instance_id, instance_wrapper in instances.items():
                if isinstance(instance_wrapper, dict):
                    for variant_name, instance_data in instance_wrapper.items():
                        if isinstance(instance_data, dict):
                            shard_assignments = instance_data.get(
                                "shardAssignments", {}
                            )
                            model_id = shard_assignments.get("modelId")
                            runner_to_shard = shard_assignments.get("runnerToShard", {})

                            if not model_id:
                                continue

                            # Check if all runners are ready
                            all_ready = True
                            for runner_id in runner_to_shard.keys():
                                runner_status = runners.get(runner_id, {})
                                if isinstance(runner_status, dict):
                                    status_type = (
                                        next(iter(runner_status.keys()))
                                        if runner_status
                                        else None
                                    )
                                    if status_type not in [
                                        "RunnerReady",
                                        "RunnerRunning",
                                    ]:
                                        all_ready = False
                                        break

                            if all_ready and runner_to_shard:
                                logger.info(
                                    f"[{thread_name}] Auto-detected model: {model_id}"
                                )
                                return model_id

            logger.warning(f"[{thread_name}] No ready model found")
            return None

        except requests.exceptions.Timeout:
            logger.error(f"[{thread_name}] Timeout querying Exo state")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"[{thread_name}] Connection error - is Exo running?")
            return None
        except Exception as e:
            logger.error(f"[{thread_name}] Failed to query Exo state: {e}")
            return None

    def _check_config_reload(self):
        """Check if .env has changed and reload configuration if needed"""
        old_exo_base_url = self.config.exo_base_url
        old_exo_api_key = self.config.exo_api_key

        if self.config.reload():
            logger.info("Detected .env file change, reloading configuration...")

            # Check if Exo settings changed
            if (
                self.config.exo_base_url != old_exo_base_url
                or self.config.exo_api_key != old_exo_api_key
            ):
                logger.info("Exo settings changed, reinitializing client...")
                self.client = self._init_openai_client()

            logger.info("Configuration reloaded successfully")

    # =========================================================================
    # ENRICHMENT PIPELINE METHODS
    # =========================================================================

    def decide_needs_enrichment(self, prompt: str) -> bool:
        """
        Call LLM to decide if prompt needs context enrichment.

        Returns True if enrichment is recommended, False otherwise.
        On error, returns True (err on the side of enrichment).
        """
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Deciding if prompt needs enrichment...")

        try:
            model_to_use = self.config.model_name or self._get_running_model()
            if not model_to_use:
                logger.warning(
                    f"[{thread_name}] No model available, assuming enrichment needed"
                )
                return True

            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": DECISION_USER_TEMPLATE.format(prompt=prompt),
                    },
                ],
                temperature=self.config.decision_temperature,
                max_tokens=10,
                timeout=self.config.llm_timeout_seconds,
            )

            answer = response.choices[0].message.content.strip().upper()
            needs_enrichment = answer.startswith("YES")

            logger.info(
                f"[{thread_name}] Enrichment decision: {answer} -> {needs_enrichment}"
            )
            return needs_enrichment

        except Exception as e:
            logger.error(f"[{thread_name}] Error in enrichment decision: {e}")
            # On error, assume enrichment is needed
            return True

    def query_chromadb(self, prompt: str) -> list[dict]:
        """
        Embed prompt and query ChromaDB for relevant chunks.

        Returns list of dicts with 'text', 'metadata', and 'distance'.
        Returns empty list on error (triggers pass-through).
        """
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Querying ChromaDB for relevant context...")

        if self.collection is None:
            logger.warning(f"[{thread_name}] ChromaDB collection not available")
            return []

        try:
            # Embed the query
            query_embedding = self.embedder.embed_query(prompt)

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.chromadb_n_results,
            )

            chunks = []
            if results["documents"] and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunks.append(
                        {"text": doc, "metadata": metadata, "distance": distance}
                    )

            logger.info(f"[{thread_name}] Retrieved {len(chunks)} chunks from ChromaDB")
            for i, chunk in enumerate(chunks[:3]):  # Log first 3
                logger.debug(
                    f"[{thread_name}]   Chunk {i + 1}: distance={chunk['distance']:.4f}, "
                    f"source={chunk['metadata'].get('source', 'unknown')}"
                )

            return chunks

        except Exception as e:
            logger.error(f"[{thread_name}] ChromaDB query failed: {e}")
            return []

    def store_contexts(self, prompt_id: str, chunks: list[dict]) -> tuple[int, int]:
        """
        Store retrieved context chunks in context.db.

        Args:
            prompt_id: ID of the prompt these contexts belong to
            chunks: List of chunks from ChromaDB query

        Returns:
            Tuple of (number of contexts stored, number of duplicates skipped)
        """
        thread_name = threading.current_thread().name
        logger.info(
            f"[{thread_name}] Storing {len(chunks)} context chunks for prompt {prompt_id}..."
        )

        if not chunks:
            logger.warning(f"[{thread_name}] No chunks to store")
            return 0, 0

        try:
            context_ids, skipped = self.context_store.add_contexts_batch(
                prompt_id, chunks
            )
            if skipped > 0:
                logger.info(
                    f"[{thread_name}] Stored {len(context_ids)} contexts ({skipped} duplicates skipped)"
                )
            else:
                logger.info(
                    f"[{thread_name}] Stored {len(context_ids)} contexts in database"
                )
            return len(context_ids), skipped

        except Exception as e:
            logger.error(f"[{thread_name}] Failed to store contexts: {e}")
            return 0, 0

    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================

    def _extract_prompt_and_id(self, message) -> tuple[str, str]:
        """Extract prompt text and ID from message"""
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                prompt = parsed.get("text", "") or parsed.get("prompt", "")
                message_id = parsed.get(
                    "id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
            except json.JSONDecodeError:
                prompt = message
                message_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        else:
            prompt = message.get("text", "") or message.get("prompt", "")
            message_id = message.get("id", datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

        return prompt, message_id

    def _produce_message(
        self,
        prompt: str,
        message_id: str,
        original_id: str,
        context_available: bool,
        contexts_stored: int = 0,
    ):
        """Produce message to output topic"""
        thread_name = threading.current_thread().name

        output_message = {
            "prompt": prompt,
            "id": message_id,
            "original_id": original_id,
            "context_available": context_available,
            "contexts_stored": contexts_stored,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            self.producer.send(self.config.kafka_output_topic, value=output_message)
            self.producer.flush()
            logger.info(
                f"[{thread_name}] Produced message to '{self.config.kafka_output_topic}'"
            )
            logger.info(
                f"[{thread_name}]   context_available={context_available}, contexts_stored={contexts_stored}"
            )
        except Exception as e:
            logger.error(f"[{thread_name}] Failed to produce message: {e}")
            raise

    def _notify_broca(self, message: str):
        """Send a status message directly to Broca (TTS) for user feedback"""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Sending status message to Broca: {message}")

        try:
            # Broca expects {"text": "...", "id": "..."} format
            broca_message = {
                "text": message,
                "id": f"enrichener_status_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            }
            self.producer.send(self.config.kafka_broca_topic, value=broca_message)
            self.producer.flush()
            logger.info(
                f"[{thread_name}] Status message sent to '{self.config.kafka_broca_topic}'"
            )
        except Exception as e:
            logger.error(f"[{thread_name}] Failed to send status message to Broca: {e}")

    def process_message(self, message):
        """Process a single message through the context retrieval pipeline"""
        thread_name = threading.current_thread().name
        logger.info("=" * 60)
        logger.info(f"[{thread_name}] STARTING MESSAGE PROCESSING")
        logger.info("=" * 60)

        try:
            # Step 1: Parse message
            logger.info(f"[{thread_name}] STEP 1/4: Parsing message...")
            prompt, message_id = self._extract_prompt_and_id(message)

            if not prompt:
                logger.warning(f"[{thread_name}] Empty prompt, skipping")
                return

            logger.info(f"[{thread_name}]   Message ID: {message_id}")
            logger.info(
                f"[{thread_name}]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
            )

            # Notify user via Broca that context retrieval is starting
            self._notify_broca("Retrieving context, one moment...")

            # Step 2: Decide if context retrieval is needed
            logger.info(
                f"[{thread_name}] STEP 2/4: Checking if context retrieval needed..."
            )
            try:
                needs_context = self.decide_needs_enrichment(prompt)
            except Exception as e:
                logger.error(f"[{thread_name}] Context decision failed: {e}")
                needs_context = False  # Pass through on error

            if not needs_context:
                logger.info(f"[{thread_name}] Context not needed, forwarding prompt...")
                self._produce_message(prompt, message_id, message_id, False)
                return

            # Step 3: Query ChromaDB
            logger.info(f"[{thread_name}] STEP 3/4: Querying ChromaDB...")
            try:
                chunks = self.query_chromadb(prompt)
            except Exception as e:
                logger.error(f"[{thread_name}] ChromaDB query failed: {e}")
                chunks = []

            if not chunks:
                logger.info(
                    f"[{thread_name}] No relevant chunks found, forwarding prompt..."
                )
                self._produce_message(prompt, message_id, message_id, False)
                return

            # Step 4: Store contexts in database and forward original prompt
            logger.info(
                f"[{thread_name}] STEP 4/4: Storing contexts and forwarding prompt..."
            )
            contexts_stored, duplicates_skipped = self.store_contexts(
                message_id, chunks
            )

            if duplicates_skipped > 0:
                logger.info(
                    f"[{thread_name}] Stored {contexts_stored} contexts ({duplicates_skipped} duplicates skipped), forwarding original prompt"
                )
            else:
                logger.info(
                    f"[{thread_name}] Stored {contexts_stored} contexts, forwarding original prompt"
                )
            self._produce_message(prompt, message_id, message_id, True, contexts_stored)

            logger.info("=" * 60)
            logger.info(f"[{thread_name}] MESSAGE PROCESSING COMPLETE")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(
                f"[{thread_name}] Unexpected error in process_message: {e}",
                exc_info=True,
            )
            # Try to pass through on any unexpected error
            try:
                prompt, message_id = self._extract_prompt_and_id(message)
                if prompt:
                    self._produce_message(prompt, message_id, message_id, False)
            except Exception:
                logger.error(f"[{thread_name}] Failed to pass through after error")

    def _process_message_wrapper(self, message_value, message_count, message_metadata):
        """Thread wrapper for message processing"""
        thread_name = threading.current_thread().name
        start_time = datetime.now()

        logger.info(f"[{thread_name}] Processing message #{message_count}")
        logger.info(f"[{thread_name}]   Topic: {message_metadata.get('topic')}")
        logger.info(f"[{thread_name}]   Partition: {message_metadata.get('partition')}")
        logger.info(f"[{thread_name}]   Offset: {message_metadata.get('offset')}")

        try:
            self.process_message(message_value)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(
                f"[{thread_name}] Successfully processed message #{message_count} in {duration:.2f}s"
            )
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(
                f"[{thread_name}] Failed to process message #{message_count}: {e}",
                exc_info=True,
            )

    def _cleanup_completed_futures(self):
        """Remove completed futures from the active list"""
        before_count = len(self.active_futures)
        self.active_futures = [f for f in self.active_futures if not f.done()]
        after_count = len(self.active_futures)
        cleaned = before_count - after_count
        if cleaned > 0:
            logger.info(
                f"[CLEANUP] Removed {cleaned} completed futures ({before_count} -> {after_count})"
            )

    # =========================================================================
    # MAIN CONSUMER LOOP
    # =========================================================================

    def start(self):
        """Start consuming messages"""
        logger.info("=" * 60)
        logger.info("STARTING ENRICHENER CONSUMER")
        logger.info("=" * 60)
        logger.info(f"Consumer topic: {self.config.kafka_input_topic}")
        logger.info(f"Producer topic: {self.config.kafka_output_topic}")
        logger.info(f"Consumer group: {self.config.kafka_consumer_group}")
        logger.info("Using threaded processing to prevent Kafka rebalancing")
        logger.info("Waiting for prompts...")
        logger.info("=" * 60)

        message_count = 0

        try:
            for message in self.consumer:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    logger.info("[MAIN THREAD] Shutdown event detected, stopping")
                    break

                message_count += 1
                receive_timestamp = datetime.now()

                logger.info(f"\n[MAIN THREAD] MESSAGE #{message_count} RECEIVED")
                logger.info(f"[MAIN THREAD]   Time: {receive_timestamp.isoformat()}")
                logger.info(f"[MAIN THREAD]   Topic: {message.topic}")
                logger.info(f"[MAIN THREAD]   Partition: {message.partition}")
                logger.info(f"[MAIN THREAD]   Offset: {message.offset}")
                logger.info(
                    f"[MAIN THREAD]   Value preview: {str(message.value)[:100]}..."
                )

                # Check for config changes
                self._check_config_reload()

                # Submit to thread pool
                message_metadata = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                }

                future = self.executor.submit(
                    self._process_message_wrapper,
                    message.value,
                    message_count,
                    message_metadata,
                )
                self.active_futures.append(future)

                logger.info(
                    f"[MAIN THREAD] Submitted message #{message_count} to thread pool"
                )
                logger.info(f"[MAIN THREAD] Active workers: {len(self.active_futures)}")

                # Periodic cleanup
                if message_count % 10 == 0:
                    self._cleanup_completed_futures()

        except KeyboardInterrupt:
            logger.info("[MAIN THREAD] Keyboard interrupt received")
        except Exception as e:
            logger.error(f"[MAIN THREAD] Fatal error: {e}", exc_info=True)
            raise
        finally:
            logger.info("[SHUTDOWN] Starting graceful shutdown...")

            # Signal shutdown
            self.shutdown_event.set()

            # Wait for in-flight messages
            in_flight = len(self.active_futures)
            if in_flight > 0:
                logger.info(
                    f"[SHUTDOWN] Waiting for {in_flight} in-flight message(s)..."
                )
                for i, future in enumerate(self.active_futures, 1):
                    try:
                        future.result(timeout=30)
                        logger.info(f"[SHUTDOWN] Message {i}/{in_flight} completed")
                    except Exception as e:
                        logger.error(f"[SHUTDOWN] Message {i}/{in_flight} failed: {e}")

            # Shutdown thread pool
            logger.info("[SHUTDOWN] Shutting down thread pool...")
            self.executor.shutdown(wait=True, cancel_futures=False)

            # Close Kafka connections
            logger.info("[SHUTDOWN] Closing Kafka connections...")
            self.consumer.close()
            self.producer.flush()
            self.producer.close()

            logger.info("[SHUTDOWN] Enrichener shutdown complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ENRICHENER APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        config = load_config()
        enricher = PromptEnricher(config)
        enricher.start()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR DURING STARTUP")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("ENRICHENER APPLICATION EXITING")
    logger.info("=" * 60)
