#!/usr/bin/env python3
# consumer.py
from kafka import KafkaConsumer, KafkaProducer
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import re

# Add parent directory to path for libs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.config import Config
from libs.inference import get_inference_client
from libs.olorin_logging import OlorinLogger
from libs.context_store import ContextStore
from libs.chat_store import ChatStore
from libs.persistent_log import get_persistent_log
from libs.state import get_state
from libs.tool_client import ToolClient

# Initialize config with hot-reload support
config = Config(watch=True)

# Set up logging
default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
log_dir = config.get("LOG_DIR", default_log_dir)
log_file = os.path.join(log_dir, "cortex-consumer.log")
env_log_level = config.get("LOG_LEVEL", "INFO")

# Initialize logger
logger = OlorinLogger(log_file=log_file, log_level=env_log_level, name=__name__)


class CortexConfig:
    """Configuration wrapper for Cortex consumer"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        """Load configuration values from Config"""
        self.kafka_bootstrap_servers = self.cfg.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.kafka_input_topic = self.cfg.get("CORTEX_INPUT_TOPIC", "prompts")
        self.kafka_output_topic = self.cfg.get("CORTEX_OUTPUT_TOPIC", "ai_out")
        self.kafka_consumer_group = self.cfg.get(
            "CORTEX_CONSUMER_GROUP", "exo-consumer-group"
        )
        self.kafka_auto_offset_reset = self.cfg.get(
            "CORTEX_AUTO_OFFSET_RESET", "earliest"
        )
        self.exo_base_url = self.cfg.get("EXO_BASE_URL", "http://localhost:52415/v1")
        self.exo_api_key = self.cfg.get("EXO_API_KEY", "dummy-key")

        # MODEL_NAME is optional - if empty/not set, will auto-detect from running instance
        model_name_env = self.cfg.get("MODEL_NAME", "").strip()
        self.model_name = model_name_env if model_name_env else None

        self.temperature = self.cfg.get_float("TEMPERATURE", 0.7)

        # MAX_TOKENS is optional - if empty/not set, will use exo's default
        max_tokens_env = self.cfg.get("MAX_TOKENS", "").strip()
        self.max_tokens = int(max_tokens_env) if max_tokens_env else None

        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

        # Context store settings (shared with hippocampus enrichener)
        # Path resolved via config library against project root
        self.context_db_path = self.cfg.get_path(
            "CONTEXT_DB_PATH", "./hippocampus/data/context.db"
        )
        self.cleanup_context_after_use = self.cfg.get_bool(
            "CLEANUP_CONTEXT_AFTER_USE", True
        )

        # Chat history settings
        # Path resolved via config library against project root
        self.chat_db_path = self.cfg.get_path("CHAT_DB_PATH", "./cortex/data/chat.db")
        self.chat_history_enabled = self.cfg.get_bool("CHAT_HISTORY_ENABLED", True)

        # Reset command patterns (comma-separated list)
        reset_patterns_str = self.cfg.get(
            "CHAT_RESET_PATTERNS",
            "/reset,reset conversation,new conversation,start over",
        )
        self.chat_reset_patterns = [
            p.strip().lower() for p in reset_patterns_str.split(",")
        ]

    def reload(self) -> bool:
        """Check for config changes and reload if needed"""
        if self.cfg.reload():
            self._load()
            return True
        return False


def load_config() -> CortexConfig:
    """Load configuration from environment variables"""
    logger.info("Loading configuration from environment variables...")

    cortex_cfg = CortexConfig(config)

    # Update logging level
    import logging

    logger.setLevel(getattr(logging, cortex_cfg.log_level.upper(), logging.INFO))

    logger.info("Configuration loaded successfully:")
    logger.info(f"  Kafka Bootstrap Servers: {cortex_cfg.kafka_bootstrap_servers}")
    logger.info(f"  Kafka Input Topic: {cortex_cfg.kafka_input_topic}")
    logger.info(f"  Kafka Output Topic: {cortex_cfg.kafka_output_topic}")
    logger.info(f"  Kafka Consumer Group: {cortex_cfg.kafka_consumer_group}")
    logger.info(f"  Kafka Auto Offset Reset: {cortex_cfg.kafka_auto_offset_reset}")
    logger.info(f"  Exo Base URL: {cortex_cfg.exo_base_url}")
    logger.info(f"  Exo API Key: {'***' if cortex_cfg.exo_api_key else 'None'}")
    logger.info(
        f"  Model Name: {cortex_cfg.model_name if cortex_cfg.model_name else 'auto-detect'}"
    )
    logger.info(f"  Temperature: {cortex_cfg.temperature}")
    logger.info(
        f"  Max Tokens: {cortex_cfg.max_tokens if cortex_cfg.max_tokens else 'use exo default'}"
    )
    logger.info(f"  Log Level: {cortex_cfg.log_level}")
    logger.info(f"  Context DB Path: {cortex_cfg.context_db_path}")
    logger.info(f"  Cleanup Context After Use: {cortex_cfg.cleanup_context_after_use}")
    logger.info(f"  Chat History Enabled: {cortex_cfg.chat_history_enabled}")
    logger.info(f"  Chat DB Path: {cortex_cfg.chat_db_path}")
    logger.info(f"  Chat Reset Patterns: {cortex_cfg.chat_reset_patterns}")

    return cortex_cfg


def extract_thinking_blocks(text: str) -> tuple[str, list[str]]:
    """
    Extract thinking blocks from text and return cleaned text + thinking block list.
    Handles both <think> and <thinking> tags.

    Returns:
        tuple: (cleaned_text, list_of_thinking_blocks)
    """
    # Pattern to match both <think>...</think> and <thinking>...</thinking> blocks
    thinking_pattern = r"<think(?:ing)?>.*?</think(?:ing)?>"

    # Find all thinking blocks
    thinking_blocks = re.findall(thinking_pattern, text, re.DOTALL)

    # Remove thinking blocks from text
    cleaned_text = re.sub(thinking_pattern, "", text, flags=re.DOTALL)

    # Clean up any excessive whitespace left behind
    cleaned_text = re.sub(r"\n\n\n+", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text, thinking_blocks


def compute_thinking_stats(thinking_blocks: list[str]) -> dict:
    """
    Compute statistics about thinking blocks.
    Handles both <think> and <thinking> tags.

    Returns:
        dict: Statistics including word count, character count, etc.
    """
    if not thinking_blocks:
        return {}

    # Concatenate all thinking blocks
    full_thinking = "\n".join(thinking_blocks)

    # Strip the tags for content analysis (handles both <think> and <thinking>)
    content_only = re.sub(r"</?think(?:ing)?>", "", full_thinking)

    # Compute stats
    char_count = len(content_only)
    word_count = len(content_only.split())
    line_count = len(content_only.strip().split("\n"))
    block_count = len(thinking_blocks)

    return {
        "block_count": block_count,
        "total_characters": char_count,
        "total_words": word_count,
        "total_lines": line_count,
        "avg_words_per_block": word_count / block_count if block_count > 0 else 0,
    }


def strip_markdown(text: str) -> str:
    """
    Strip markdown formatting from text for TTS readability.

    Removes:
    - Code blocks (```...```)
    - Inline code (`...`)
    - Bold/italic markers (**, __, *, _)
    - Headers (#, ##, etc.)
    - Links [text](url) - keeps text only
    - Lists markers (-, *, 1., etc.)
    - Blockquotes (>)
    - Horizontal rules
    - HTML tags

    Returns:
        str: Plain text suitable for TTS
    """
    # Remove code blocks (triple backticks with optional language)
    text = re.sub(r"```[a-z]*\n.*?\n```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Convert links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove images ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

    # Remove bold/italic markers (process longer patterns first to avoid partial matches)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # Bold+italic ***text***
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold **text**
    text = re.sub(r"___(.+?)___", r"\1", text)  # Bold+italic ___text___
    text = re.sub(r"__(.+?)__", r"\1", text)  # Bold __text__
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text
    )  # Italic *text* (not part of **)
    text = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", text
    )  # Italic _text_ (not part of __)

    # Remove any remaining stray asterisks (handles malformed markdown, standalone *, **, etc.)
    text = re.sub(r"\*+", "", text)

    # Remove headers (# Header) - keep the text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove list markers but keep the text
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # Unordered lists
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # Ordered lists

    # Clean up excessive whitespace
    text = re.sub(r"\n\n\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

    return text


class ExoConsumer:
    def __init__(self, config: CortexConfig):
        logger.info("Initializing ExoConsumer...")
        self.config = config

        # Initialize Kafka consumer
        logger.info(
            f"Creating Kafka consumer for topic '{config.kafka_input_topic}'..."
        )
        logger.info(f"  Bootstrap servers: {config.kafka_bootstrap_servers}")
        logger.info(f"  Consumer group: {config.kafka_consumer_group}")
        logger.info(f"  Auto offset reset: {config.kafka_auto_offset_reset}")
        try:
            self.consumer = KafkaConsumer(
                config.kafka_input_topic,
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_deserializer=lambda m: m.decode("utf-8"),
                auto_offset_reset=config.kafka_auto_offset_reset,
                enable_auto_commit=True,
                group_id=config.kafka_consumer_group,
                # Increase timeouts to prevent rebalancing during long AI inference
                # max_poll_interval_ms: Maximum time between polls before consumer is considered dead
                max_poll_interval_ms=600000,  # 10 minutes (default is 5 minutes)
                # session_timeout_ms: Maximum time between heartbeats before consumer is considered dead
                session_timeout_ms=60000,  # 60 seconds (default is 10 seconds)
                # heartbeat_interval_ms: How often to send heartbeats
                heartbeat_interval_ms=10000,  # 10 seconds
            )
            logger.info("Kafka consumer created successfully")
            logger.info("  max_poll_interval_ms: 600000 (10 minutes)")
            logger.info("  session_timeout_ms: 60000 (60 seconds)")
            logger.info("  heartbeat_interval_ms: 10000 (10 seconds)")
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}", exc_info=True)
            raise

        # Initialize Kafka producer for output
        logger.info(
            f"Creating Kafka producer for output topic '{config.kafka_output_topic}'..."
        )
        logger.info(f"  Bootstrap servers: {config.kafka_bootstrap_servers}")
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka producer created successfully")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}", exc_info=True)
            raise

        # Initialize inference client (centralized LLM access)
        self.inference = get_inference_client()
        logger.info(
            f"Inference client initialized (backend: {self.inference.backend_type.value})"
        )

        # Cache model capabilities for context validation
        self._model_capabilities = None  # Will be populated on first use

        # Initialize context store for reading enriched contexts
        self._init_context_store()

        # Initialize chat store for conversation history
        if self.config.chat_history_enabled:
            self._init_chat_store()
        else:
            self.chat_store = None
            logger.info("Chat history disabled by configuration")

        # Initialize thread pool for message processing
        # Using max_workers=3 to handle multiple prompts concurrently if needed
        self.executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="msg-processor"
        )
        self.active_futures = []
        self.shutdown_event = threading.Event()
        logger.info("Thread pool executor initialized with 3 workers")

        # Tool call support detection
        self._tools_supported: bool | None = None  # None = not yet detected
        self._state = (
            get_state()
        )  # Singleton state manager for cross-component visibility

        # AI Tool client for tool use (separate from slash commands)
        self.tool_client = ToolClient(self.config.cfg)
        self.available_tools: list[dict] = []  # Will be populated in start()

        logger.info("ExoConsumer initialized successfully")
        logger.info(f"  Input topic: {config.kafka_input_topic}")
        logger.info(f"  Output topic: {config.kafka_output_topic}")
        logger.info(f"  Exo endpoint: {config.exo_base_url}")
        logger.info(f"  Model: {config.model_name}")

    def check_tool_support(self, force_redetect: bool = False) -> bool:
        """
        Check if tool calls are supported by the current model/endpoint.

        Results are cached in memory and persisted to state.db for visibility
        in the olorin-inspector and other components.

        Args:
            force_redetect: If True, bypass cache and re-run detection

        Returns:
            bool: True if tool calls are supported
        """
        thread_name = threading.current_thread().name

        # Return cached result if available and not forcing redetection
        if self._tools_supported is not None and not force_redetect:
            logger.debug(
                f"[{thread_name}] Using cached tool support: {self._tools_supported}"
            )
            return self._tools_supported

        # Delegate detection to centralized inference client
        self._tools_supported = self.inference.supports_tools()

        # Store result in state.db for cross-component visibility
        try:
            model_name = self.inference.get_running_model() or "unknown"
            self._state.set_bool("cortex.tools_supported", self._tools_supported)
            self._state.set_string("cortex.tools_model", model_name)
            self._state.set_string(
                "cortex.tools_checked_at", datetime.now().isoformat()
            )

            logger.info(f"[{thread_name}] Stored tool support status in state.db:")
            logger.info(
                f"[{thread_name}]   cortex.tools_supported = {self._tools_supported}"
            )
            logger.info(f"[{thread_name}]   cortex.tools_model = {model_name}")
        except Exception as e:
            logger.warning(
                f"[{thread_name}] Failed to store tool support in state.db: {e}"
            )

        return self._tools_supported

    def check_model_capabilities(self, model: str | None = None) -> None:
        """
        Check and cache model capabilities, logging warnings for problematic configurations.

        This detects issues like sliding window attention that can cause context
        to be silently truncated, leading to confusing "I don't have that information"
        responses even when context was provided.

        Args:
            model: Model to check (uses running model if None)
        """
        thread_name = threading.current_thread().name

        capabilities = self.inference.get_model_capabilities(model)
        if capabilities is None:
            logger.warning(
                f"[{thread_name}] Could not retrieve model capabilities - "
                "context validation will be skipped"
            )
            return

        self._model_capabilities = capabilities

        # Store capabilities in state.db for visibility
        try:
            self._state.set_string("cortex.model_id", capabilities.model_id)
            self._state.set_int("cortex.context_length", capabilities.context_length)
            if capabilities.sliding_window:
                self._state.set_int(
                    "cortex.sliding_window", capabilities.sliding_window
                )
            else:
                self._state.delete("cortex.sliding_window")
            self._state.set_int(
                "cortex.effective_context", capabilities.effective_context
            )
        except Exception as e:
            logger.warning(
                f"[{thread_name}] Failed to store model capabilities in state.db: {e}"
            )

        # Log capability summary
        logger.info(f"[{thread_name}] Model capabilities for {capabilities.model_id}:")
        logger.info(
            f"[{thread_name}]   Context length: {capabilities.context_length:,} tokens"
        )

        if capabilities.has_sliding_window:
            logger.error(
                f"[{thread_name}] ⚠️ CRITICAL: Model uses SLIDING WINDOW ATTENTION "
                f"({capabilities.sliding_window:,} tokens)!"
            )
            logger.error(
                f"[{thread_name}]   This model can only 'see' the last "
                f"{capabilities.sliding_window:,} tokens when generating responses."
            )
            logger.error(
                f"[{thread_name}]   RAG context and conversation history beyond this "
                "window will be INVISIBLE to the model."
            )
            logger.error(
                f"[{thread_name}]   Consider using a model with full attention for "
                "RAG/long context use cases."
            )
        else:
            logger.info(f"[{thread_name}]   Attention: Full (no sliding window)")
            logger.info(
                f"[{thread_name}]   Effective context: {capabilities.effective_context:,} tokens"
            )

    def validate_message_context(
        self, messages: list[dict], model: str | None = None
    ) -> tuple[bool, str | None]:
        """
        Validate that messages fit within the model's context window.

        Args:
            messages: List of message dicts to validate
            model: Model to check against (uses cached capabilities if None)

        Returns:
            Tuple of (fits: bool, warning_message: Optional[str])
        """
        # Use cached capabilities or fetch if needed
        if self._model_capabilities is None:
            self.check_model_capabilities(model)

        if self._model_capabilities is None:
            return (True, None)  # Can't validate, assume OK

        return self._model_capabilities.check_context_fit(messages)

    def _init_context_store(self):
        """Initialize context store for reading enriched contexts from hippocampus"""
        logger.info(f"Initializing context store at {self.config.context_db_path}...")
        try:
            self.context_store = ContextStore(self.config.context_db_path)
            stats = self.context_store.get_statistics()
            logger.info("Context store initialized successfully")
            logger.info(f"  Existing contexts: {stats['total_contexts']}")
            logger.info(f"  Unique prompts: {stats['unique_prompts']}")
        except Exception as e:
            logger.warning(f"Failed to initialize context store: {e}")
            logger.warning("Context enrichment will be disabled")
            self.context_store = None

    def _get_manual_context_documents(self) -> list[dict]:
        """
        Get manually selected context documents from context_documents table.

        These are documents the user manually added via the chat search interface.
        They persist until the user removes them (they are not auto-cleared).

        Returns:
            List of context dicts in the same format as ContextStore:
            {'content': str, 'source': str|None, 'id': str, ...}
        """
        thread_name = threading.current_thread().name

        if not self.config.context_db_path:
            return []

        try:
            import sqlite3

            # Use the same context.db but different table
            conn = sqlite3.connect(self.config.context_db_path)
            conn.row_factory = sqlite3.Row

            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='context_documents'"
            )
            if cursor.fetchone() is None:
                conn.close()
                return []

            # Fetch all manual context documents
            cursor = conn.execute(
                """
                SELECT id, text, source, added_at
                FROM context_documents
                ORDER BY added_at ASC
                """
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            # Convert to ContextStore-compatible format
            manual_contexts = []
            for row in rows:
                manual_contexts.append(
                    {
                        "id": row["id"],
                        "content": row["text"],
                        "source": row["source"],
                        "h1": None,
                        "h2": None,
                        "h3": None,
                        "chunk_index": None,
                        "distance": None,  # Manual context has no distance score
                        "added_at": row["added_at"],
                        "is_manual": True,  # Flag to distinguish from auto context
                    }
                )

            logger.info(
                f"[{thread_name}] Retrieved {len(manual_contexts)} manual context document(s)"
            )
            return manual_contexts

        except Exception as e:
            logger.warning(f"[{thread_name}] Failed to get manual context: {e}")
            return []

    def _init_chat_store(self):
        """Initialize chat store for conversation history management"""
        logger.info(f"Initializing chat store at {self.config.chat_db_path}...")
        try:
            self.chat_store = ChatStore(self.config.chat_db_path)
            stats = self.chat_store.get_statistics()
            logger.info("Chat store initialized successfully")
            logger.info(f"  Total conversations: {stats['total_conversations']}")
            logger.info(f"  Active conversations: {stats['active_conversations']}")
            logger.info(f"  Total messages: {stats['total_messages']}")
        except Exception as e:
            logger.warning(f"Failed to initialize chat store: {e}")
            logger.warning("Chat history will be disabled")
            self.chat_store = None

    def _start_processing_notices(
        self, cancel_event: threading.Event, message_id: str, thread_name: str
    ) -> threading.Thread:
        """
        Start a background thread that sends periodic "Processing" notices.

        The first notice is sent immediately, then subsequent notices are sent
        with increasing delays: 10s -> 12s -> 14.4s -> ... up to max 20s.
        The delay grows by 20% each time.

        Args:
            cancel_event: Event to signal when to stop sending notices
            message_id: The message ID for tagging notices
            thread_name: Thread name for logging

        Returns:
            The background thread (already started)
        """

        def notice_sender():
            notice_count = 0
            delay = 10.0  # Initial delay in seconds
            max_delay = 20.0
            growth_factor = 1.2  # 20% increase

            # Send first notice immediately
            notice_count += 1
            processing_message = {
                "text": "Processing, one moment...",
                "id": f"{message_id}_processing_{notice_count}",
                "prompt_id": message_id,
                "is_processing_notice": True,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"[{thread_name}] 📣 Sending processing notice #{notice_count}")
            self.producer.send(self.config.kafka_output_topic, value=processing_message)

            # Continue sending notices with increasing delays until cancelled
            while not cancel_event.is_set():
                # Wait for the delay, but check for cancellation
                if cancel_event.wait(timeout=delay):
                    # Event was set, stop sending notices
                    logger.info(
                        f"[{thread_name}] ✓ Processing notices cancelled after {notice_count} notice(s)"
                    )
                    return

                # Send another notice
                notice_count += 1
                processing_message = {
                    "text": "Processing, please wait...",
                    "id": f"{message_id}_processing_{notice_count}",
                    "prompt_id": message_id,
                    "is_processing_notice": True,
                    "timestamp": datetime.now().isoformat(),
                }
                logger.info(
                    f"[{thread_name}] 📣 Sending processing notice #{notice_count} (next in {min(delay * growth_factor, max_delay):.1f}s)"
                )
                self.producer.send(
                    self.config.kafka_output_topic, value=processing_message
                )

                # Increase delay for next time, capped at max
                delay = min(delay * growth_factor, max_delay)

        thread = threading.Thread(
            target=notice_sender, name=f"{thread_name}-notice-sender", daemon=True
        )
        thread.start()
        return thread

    def _is_reset_command(self, prompt: str) -> bool:
        """Check if the prompt is a conversation reset command."""
        prompt_lower = prompt.strip().lower()
        for pattern in self.config.chat_reset_patterns:
            if prompt_lower == pattern or prompt_lower.startswith(pattern + " "):
                return True
        return False

    def _handle_reset_command(self, message_id: str) -> bool:
        """
        Handle a conversation reset command.
        Returns True if reset was handled (should skip normal processing).
        """
        thread_name = threading.current_thread().name

        if self.chat_store is None:
            logger.info(f"[{thread_name}] Reset requested but chat history is disabled")
            return False

        try:
            new_conv_id = self.chat_store.reset_conversation()
            logger.info(
                f"[{thread_name}] Conversation reset. New conversation ID: {new_conv_id}"
            )

            # Send confirmation to output topic for Broca to announce
            reset_message = {
                "text": "Conversation has been reset. Starting fresh.",
                "id": f"{message_id}_reset_confirmation",
                "prompt_id": message_id,
                "is_reset_confirmation": True,
                "new_conversation_id": new_conv_id,
                "timestamp": datetime.now().isoformat(),
            }
            self.producer.send(self.config.kafka_output_topic, value=reset_message)
            return True

        except Exception as e:
            logger.error(f"[{thread_name}] Failed to reset conversation: {e}")
            return False

    def _format_context_as_exchange(self, context_chunks: list[dict]) -> list[dict]:
        """
        Format RAG context chunks as a user/assistant exchange for injection.

        Returns two messages:
        1. User message presenting the context
        2. Assistant acknowledgment

        Args:
            context_chunks: List of context dicts with 'content', 'source', etc.
                           May include 'is_manual' flag to distinguish manually selected context.

        Returns:
            List of two message dicts [user_msg, assistant_msg]
        """
        # Separate manual and auto context for clearer presentation
        manual_chunks = [c for c in context_chunks if c.get("is_manual")]
        auto_chunks = [c for c in context_chunks if not c.get("is_manual")]

        context_parts = []

        # Format manual context first (user-selected)
        if manual_chunks:
            context_parts.append("## Selected Context (manually chosen)")
            for i, ctx in enumerate(manual_chunks):
                source_parts = []
                if ctx.get("source"):
                    source_parts.append(ctx["source"])
                source_ref = (
                    " > ".join(source_parts)
                    if source_parts
                    else f"Selected document {i + 1}"
                )
                context_parts.append(f"### {source_ref}\n{ctx['content']}")

        # Format auto context (RAG-retrieved)
        if auto_chunks:
            if manual_chunks:
                context_parts.append("\n## Retrieved Context (automatically found)")
            for i, ctx in enumerate(auto_chunks):
                source_parts = []
                if ctx.get("source"):
                    source_parts.append(ctx["source"])
                if ctx.get("h1"):
                    source_parts.append(ctx["h1"])
                if ctx.get("h2"):
                    source_parts.append(ctx["h2"])
                if ctx.get("h3"):
                    source_parts.append(ctx["h3"])

                source_ref = (
                    " > ".join(source_parts) if source_parts else f"Source {i + 1}"
                )
                context_parts.append(f"### {source_ref}\n{ctx['content']}")

        context_block = "\n\n".join(context_parts)

        user_context_message = f"""Based on my knowledge base, here is relevant context that may help answer my upcoming question:

---
{context_block}
---"""

        assistant_ack = (
            "I understand. I'll use this context to help answer your questions."
        )

        return [
            {"role": "user", "content": user_context_message},
            {"role": "assistant", "content": assistant_ack},
        ]

    def _build_tool_system_prompt(self) -> str | None:
        """
        Build a system prompt that describes available tools and how to use them.

        Returns:
            System prompt string if tools are available, None otherwise
        """
        if not self.available_tools:
            return None

        tool_descriptions = []
        for tool in self.available_tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter documentation
            param_docs = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                req_marker = " (required)" if param_name in required else " (optional)"
                param_docs.append(
                    f"    - {param_name} ({param_type}){req_marker}: {param_desc}"
                )

            params_str = "\n".join(param_docs) if param_docs else "    (no parameters)"

            tool_descriptions.append(f"""- **{name}**: {description}
  Parameters:
{params_str}""")

        tools_block = "\n\n".join(tool_descriptions)

        system_prompt = f"""You are a helpful AI assistant with access to tools. When the user asks you to perform an action that matches one of your available tools, you MUST use the appropriate tool to complete the task.

## Available Tools

{tools_block}

## When to Use Tools

- When the user asks you to write, save, or export content to a file, use the `write` tool.
- When the user mentions a filename or asks to create a file, use the `write` tool.
- Always use tools when they match the user's request - do not just describe what you would do.

## How to Use Tools

When you decide to use a tool, call it with the required parameters. After the tool executes, you will receive the result and can then respond to the user with confirmation or any follow-up information."""

        return system_prompt

    def _build_messages_with_history(
        self, prompt: str, message_id: str, context_chunks: list[dict] | None = None
    ) -> list[dict]:
        """
        Build the complete messages array for API call including conversation history.

        Structure:
        1. Historical user/assistant exchanges from conversation DB
        2. If context_chunks provided (RAG): inject as user/assistant exchange
        3. Current user prompt

        Args:
            prompt: The current user prompt
            message_id: Kafka message ID for this prompt
            context_chunks: Optional list of RAG context chunks

        Returns:
            List of message dicts for OpenAI API
        """
        thread_name = threading.current_thread().name
        messages = []

        # Skip tool system prompt when significant RAG context is provided
        # Small models get confused trying to use tools when they should answer from context
        # Threshold: if context is more than 10KB, prioritize RAG over tools
        context_size = sum(len(c.get("content", "")) for c in (context_chunks or []))
        skip_tools_for_rag = context_size > 10000  # 10KB threshold

        # Add system prompt for tools if available (and not doing heavy RAG)
        tool_system_prompt = None
        if not skip_tools_for_rag:
            tool_system_prompt = self._build_tool_system_prompt()
            if tool_system_prompt:
                messages.append({"role": "system", "content": tool_system_prompt})
                logger.info(
                    f"[{thread_name}] Added tool system prompt ({len(tool_system_prompt)} chars)"
                )
        else:
            logger.info(
                f"[{thread_name}] Skipping tool system prompt - RAG context is {context_size:,} chars "
                "(tools disabled for large context to prevent model confusion)"
            )

        # If chat history is disabled, just return the current prompt
        if self.chat_store is None:
            if context_chunks:
                messages.extend(self._format_context_as_exchange(context_chunks))
            messages.append({"role": "user", "content": prompt})
            logger.info(
                f"[{thread_name}] Chat history disabled, built {len(messages)} message(s)"
            )
            return messages

        try:
            # Get or create active conversation
            conversation_id = self.chat_store.get_or_create_active_conversation()
            logger.info(f"[{thread_name}] Using conversation: {conversation_id}")

            # Get historical messages
            history = self.chat_store.get_conversation_messages(conversation_id)
            logger.info(f"[{thread_name}] Retrieved {len(history)} historical messages")

            # Extract context IDs already injected in this conversation
            # This prevents re-injecting manual context that's already in history
            already_injected_ids = set()
            for msg in history:
                if msg.get("message_type") == "context_user":
                    metadata = msg.get("metadata", {})
                    if isinstance(metadata, dict):
                        ctx_ids = metadata.get("context_ids", [])
                        if ctx_ids:
                            already_injected_ids.update(ctx_ids)

            if already_injected_ids:
                logger.info(
                    f"[{thread_name}] Found {len(already_injected_ids)} already-injected context IDs"
                )

            # Add historical messages
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Filter context_chunks to only include NEW context (not already in history)
            new_context_chunks = None
            if context_chunks:
                new_context_chunks = [
                    ctx
                    for ctx in context_chunks
                    if ctx.get("id") not in already_injected_ids
                ]
                filtered_count = len(context_chunks) - len(new_context_chunks)
                if filtered_count > 0:
                    logger.info(
                        f"[{thread_name}] Filtered out {filtered_count} already-injected context chunks"
                    )

            # Add context injection (only NEW context not already in history)
            if new_context_chunks:
                context_exchange = self._format_context_as_exchange(new_context_chunks)
                messages.extend(context_exchange)
                logger.info(
                    f"[{thread_name}] Injected RAG context as user/assistant exchange"
                )

                # Store context exchange in chat history so it persists for follow-up questions
                # This ensures the LLM has access to context in subsequent turns
                try:
                    # Build metadata with context chunk info for audit trail
                    # Use new_context_chunks (filtered) to track what was actually injected
                    context_metadata = {
                        "context_ids": [
                            ctx.get("id") for ctx in new_context_chunks if ctx.get("id")
                        ],
                        "sources": [
                            ctx.get("source")
                            for ctx in new_context_chunks
                            if ctx.get("source")
                        ],
                        "distances": [
                            ctx.get("distance")
                            for ctx in new_context_chunks
                            if ctx.get("distance") is not None
                        ],
                        "chunk_count": len(new_context_chunks),
                    }
                    self.chat_store.add_user_message(
                        conversation_id,
                        context_exchange[0]["content"],
                        prompt_id=f"{message_id}_context",
                        message_type="context_user",
                        metadata=context_metadata,
                    )
                    self.chat_store.add_assistant_message(
                        conversation_id,
                        context_exchange[1]["content"],
                        message_type="context_ack",
                    )
                    logger.info(
                        f"[{thread_name}] Stored context exchange in chat history for conversation continuity"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{thread_name}] Failed to store context in chat history: {e}"
                    )

            # Add current user prompt
            messages.append({"role": "user", "content": prompt})

            # Store the user message in chat history
            self.chat_store.add_user_message(
                conversation_id, prompt, prompt_id=message_id
            )

            logger.info(
                f"[{thread_name}] Built messages array with {len(messages)} total messages"
            )
            logger.info(
                f"[{thread_name}]   - System: {1 if tool_system_prompt else 0}, History: {len(history)}, Context exchange: {2 if new_context_chunks else 0}, New prompt: 1"
            )
            return messages

        except Exception as e:
            logger.error(
                f"[{thread_name}] Error building message history: {e}", exc_info=True
            )
            # Fallback: return just the current prompt with context if available
            # Re-add system prompt since we're rebuilding messages
            messages = []
            if tool_system_prompt:
                messages.append({"role": "system", "content": tool_system_prompt})
            if context_chunks:
                messages.extend(self._format_context_as_exchange(context_chunks))
            messages.append({"role": "user", "content": prompt})
            logger.warning(
                f"[{thread_name}] Fallback: built {len(messages)} message(s) without history"
            )
            return messages

    def _build_context_enriched_prompt(
        self, prompt: str, prompt_id: str
    ) -> tuple[str, int]:
        """
        Build a context-enriched prompt by prepending relevant context chunks.

        Args:
            prompt: The original user prompt
            prompt_id: The ID to look up context in context.db

        Returns:
            Tuple of (enriched_prompt, context_count)
        """
        thread_name = threading.current_thread().name

        if self.context_store is None:
            logger.warning(
                f"[{thread_name}] Context store not available, using original prompt"
            )
            return prompt, 0

        try:
            contexts = self.context_store.get_contexts_for_prompt(prompt_id)
            if not contexts:
                logger.info(
                    f"[{thread_name}] No context found for prompt_id={prompt_id}"
                )
                return prompt, 0

            logger.info(
                f"[{thread_name}] Retrieved {len(contexts)} context chunks for prompt_id={prompt_id}"
            )

            # Build context block
            context_parts = []
            for i, ctx in enumerate(contexts):
                # Build source reference from metadata
                source_parts = []
                if ctx.get("source"):
                    source_parts.append(ctx["source"])
                if ctx.get("h1"):
                    source_parts.append(ctx["h1"])
                if ctx.get("h2"):
                    source_parts.append(ctx["h2"])
                if ctx.get("h3"):
                    source_parts.append(ctx["h3"])

                source_ref = (
                    " > ".join(source_parts) if source_parts else f"Chunk {i + 1}"
                )
                distance = ctx.get("distance")
                distance_str = (
                    f" (relevance: {1 - distance:.2%})" if distance is not None else ""
                )

                context_parts.append(
                    f"### {source_ref}{distance_str}\n{ctx['content']}"
                )

            # Combine contexts with the original prompt
            context_block = "\n\n".join(context_parts)
            enriched_prompt = f"""The following context may be helpful for answering the user's question:

---
{context_block}
---

User's question: {prompt}"""

            logger.info(
                f"[{thread_name}] Built enriched prompt with {len(contexts)} context chunks"
            )
            logger.debug(
                f"[{thread_name}] Context block length: {len(context_block)} chars"
            )

            return enriched_prompt, len(contexts)

        except Exception as e:
            logger.error(
                f"[{thread_name}] Error retrieving context: {e}", exc_info=True
            )
            return prompt, 0

    def _cleanup_context(self, prompt_id: str):
        """Clean up context from database after use"""
        thread_name = threading.current_thread().name

        if self.context_store is None:
            return

        if not self.config.cleanup_context_after_use:
            logger.debug(
                f"[{thread_name}] Context cleanup disabled, keeping context for prompt_id={prompt_id}"
            )
            return

        try:
            deleted = self.context_store.delete_contexts_for_prompt(prompt_id)
            logger.info(
                f"[{thread_name}] Cleaned up {deleted} context chunks for prompt_id={prompt_id}"
            )
        except Exception as e:
            logger.warning(f"[{thread_name}] Failed to clean up context: {e}")

    def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        conversation_id: str | None,
        prompt_id: str | None,
    ) -> list[dict]:
        """
        Execute accumulated tool calls and return results.

        Args:
            tool_calls: List of tool call objects from the API response
            conversation_id: Active conversation ID for chat history
            prompt_id: Original prompt ID for tracking

        Returns:
            List of tool result message dicts for the next API call
        """
        thread_name = threading.current_thread().name
        results = []

        # Store the assistant's tool call message in chat history
        if self.chat_store is not None and conversation_id:
            try:
                # Convert tool calls to serializable format
                tool_calls_data = []
                for tc in tool_calls:
                    tool_calls_data.append(
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": tc.get("function", {}),
                        }
                    )
                self.chat_store.add_tool_call_message(
                    conversation_id, tool_calls_data, prompt_id
                )
                logger.info(
                    f"[{thread_name}] 💬 Stored tool call message in chat history"
                )
            except Exception as e:
                logger.warning(
                    f"[{thread_name}] ⚠ Failed to store tool call message: {e}"
                )

        # Execute each tool call
        for tc in tool_calls:
            tool_call_id = tc.get("id", "")
            function_info = tc.get("function", {})
            function_name = function_info.get("name", "")
            arguments_str = function_info.get("arguments", "{}")

            logger.info(
                f"[{thread_name}] 🔧 Executing tool: {function_name} (id={tool_call_id})"
            )

            try:
                # Parse arguments
                arguments = json.loads(arguments_str)
                logger.debug(f"[{thread_name}]   Arguments: {arguments}")

                # Call the tool
                result = self.tool_client.call_tool(function_name, arguments)

                if result.get("success"):
                    result_content = result.get("result", "")
                    logger.info(
                        f"[{thread_name}] ✓ Tool {function_name} succeeded: {result_content[:100]}..."
                    )
                else:
                    error = result.get("error", {})
                    result_content = f"Error: {error.get('type', 'Unknown')}: {error.get('message', 'Unknown error')}"
                    logger.warning(
                        f"[{thread_name}] ✗ Tool {function_name} failed: {result_content}"
                    )

            except json.JSONDecodeError as e:
                result_content = f"Error: Invalid arguments JSON: {e}"
                logger.error(f"[{thread_name}] ✗ Failed to parse arguments: {e}")
            except Exception as e:
                result_content = f"Error: {type(e).__name__}: {e}"
                logger.error(f"[{thread_name}] ✗ Tool execution error: {e}")

            # Store tool result in chat history
            if self.chat_store is not None and conversation_id:
                try:
                    self.chat_store.add_tool_result_message(
                        conversation_id, tool_call_id, function_name, result_content
                    )
                    logger.info(
                        f"[{thread_name}] 💬 Stored tool result in chat history"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{thread_name}] ⚠ Failed to store tool result: {e}"
                    )

            # Build the tool result message for the API
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )

        return results

    def _check_config_reload(self):
        """Check if config has changed and reload if needed"""
        logger.debug("Checking for config file changes...")

        old_model_name = self.config.model_name
        old_temperature = self.config.temperature
        old_max_tokens = self.config.max_tokens

        if self.config.reload():
            logger.info("Detected config file change, reloading configuration...")

            # Inference client handles its own reload
            if self.inference.reload():
                logger.info(
                    f"Inference client reloaded (backend: {self.inference.backend_type.value})"
                )

            # Log other important changes
            if self.config.model_name != old_model_name:
                old_model = old_model_name if old_model_name else "auto-detect"
                new_model = (
                    self.config.model_name if self.config.model_name else "auto-detect"
                )
                logger.info(f"Model changed: {old_model} -> {new_model}")
            if self.config.temperature != old_temperature:
                logger.info(
                    f"Temperature changed: {old_temperature} -> {self.config.temperature}"
                )
            if self.config.max_tokens != old_max_tokens:
                old_tokens = old_max_tokens if old_max_tokens else "default"
                new_tokens = (
                    self.config.max_tokens if self.config.max_tokens else "default"
                )
                logger.info(f"Max tokens changed: {old_tokens} -> {new_tokens}")

            # Re-detect tool support when model changes
            if self.config.model_name != old_model_name:
                logger.info("Re-detecting tool support after model change...")
                self._tools_supported = None  # Clear cache
                tools_supported = self.check_tool_support(force_redetect=True)
                # Re-discover tools if supported
                if tools_supported:
                    logger.info("Re-discovering AI tools...")
                    self.available_tools = self.tool_client.discover_tools()
                else:
                    self.available_tools = []

            logger.info("Configuration reloaded successfully")
        else:
            logger.debug("No config file changes detected")

    def process_message(self, message):
        """Process a single prompt message and send to exo"""
        thread_name = threading.current_thread().name
        logger.info("▼" * 60)
        logger.info(f"[{thread_name}] ▶▶▶ STARTING MESSAGE PROCESSING ◀◀◀")
        logger.info("▼" * 60)
        logger.info(f"[{thread_name}] Raw message: {message}")
        logger.info(f"[{thread_name}] Message type: {type(message)}")

        # Initialize thinking block state for this message
        in_thinking = False

        # Initialize processing notice event (will be set when notices start)
        processing_cancel_event = None

        try:
            # Parse message
            logger.info(f"[{thread_name}] STEP 1/6: Parsing message format...")
            context_available = False
            contexts_stored = 0
            if isinstance(message, str):
                logger.info(
                    f"[{thread_name}] Message is a string, attempting JSON parse..."
                )
                try:
                    parsed = json.loads(message)
                    logger.info(f"[{thread_name}] ✓ Successfully parsed JSON: {parsed}")
                    prompt = parsed.get("text", "") or parsed.get("prompt", "")
                    message_id = parsed.get(
                        "id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    )
                    context_available = parsed.get("context_available", False)
                    contexts_stored = parsed.get("contexts_stored", 0)
                    logger.info(f"[{thread_name}] Extracted from JSON:")
                    logger.info(f"[{thread_name}]   - Prompt: '{prompt[:50]}...'")
                    logger.info(f"[{thread_name}]   - Message ID: {message_id}")
                    logger.info(
                        f"[{thread_name}]   - Context available: {context_available}"
                    )
                    logger.info(
                        f"[{thread_name}]   - Contexts stored: {contexts_stored}"
                    )
                except json.JSONDecodeError as json_err:
                    # Treat as plain text
                    logger.warning(f"[{thread_name}] JSON decode failed: {json_err}")
                    logger.info(f"[{thread_name}] ℹ Treating as plain text message")
                    prompt = message
                    message_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    logger.info(
                        f"[{thread_name}]   - Plain text prompt: '{prompt[:50]}...'"
                    )
                    logger.info(f"[{thread_name}]   - Generated ID: {message_id}")
            else:
                logger.info(f"[{thread_name}] Message is a dict/object")
                prompt = message.get("text", "") or message.get("prompt", "")
                message_id = message.get(
                    "id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
                context_available = message.get("context_available", False)
                contexts_stored = message.get("contexts_stored", 0)
                logger.info(f"[{thread_name}]   - Prompt: '{prompt[:50]}...'")
                logger.info(f"[{thread_name}]   - Message ID: {message_id}")
                logger.info(
                    f"[{thread_name}]   - Context available: {context_available}"
                )
                logger.info(f"[{thread_name}]   - Contexts stored: {contexts_stored}")

            if not prompt:
                logger.warning(
                    f"[{thread_name}] ✗ Empty prompt detected in message: {message}"
                )
                logger.warning(
                    f"[{thread_name}] SKIPPING message processing - no content to process"
                )
                return

            logger.info(f"[{thread_name}] ✓ Message parsing complete")
            logger.info(f"[{thread_name}]   Message ID: {message_id}")
            logger.info(
                f"[{thread_name}]   Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
            )
            logger.info(f"[{thread_name}]   Prompt length: {len(prompt)} characters")

            # Log received message to persistent log
            plog = get_persistent_log()
            plog.log(
                component="cortex",
                direction="received",
                content={"prompt": prompt, "raw_message": message},
                message_id=message_id,
                topic=self.config.kafka_input_topic,
                metadata={
                    "context_available": context_available,
                    "contexts_stored": contexts_stored,
                },
            )

            # Check for conversation reset command
            if self._is_reset_command(prompt):
                logger.info(f"[{thread_name}] 🔄 Reset command detected: '{prompt}'")
                if self._handle_reset_command(message_id):
                    logger.info(
                        f"[{thread_name}] ✓ Conversation reset completed, skipping normal processing"
                    )
                    return
                logger.warning(
                    f"[{thread_name}] ⚠ Reset handling failed, continuing with normal processing"
                )

            # Step 2: Retrieve context chunks and build messages with history
            logger.info(
                f"[{thread_name}] STEP 2/6: Building messages with conversation history..."
            )

            # First, always check for manually selected context documents
            # These are documents the user added via the search interface
            manual_context = self._get_manual_context_documents()
            manual_count = len(manual_context) if manual_context else 0
            if manual_count > 0:
                logger.info(
                    f"[{thread_name}] 📋 Found {manual_count} manually selected context document(s)"
                )

            # Then check for automatic RAG context from enrichener
            auto_context = []
            auto_count = 0
            if context_available and contexts_stored > 0:
                logger.info(
                    f"[{thread_name}] 📚 Auto context available! Retrieving {contexts_stored} context chunks..."
                )
                if self.context_store is not None:
                    auto_context = self.context_store.get_contexts_for_prompt(
                        message_id
                    )
                    auto_count = len(auto_context) if auto_context else 0
                    if auto_count > 0:
                        logger.info(
                            f"[{thread_name}] ✓ Retrieved {auto_count} auto context chunks for RAG injection"
                        )
                    else:
                        logger.warning(
                            f"[{thread_name}] ⚠ Context was indicated but none retrieved"
                        )
                else:
                    logger.warning(
                        f"[{thread_name}] ⚠ Context store not available, skipping auto context retrieval"
                    )
            else:
                logger.info(
                    f"[{thread_name}] No auto context enrichment needed (context_available={context_available})"
                )

            # Combine manual and automatic context (manual first, then auto)
            context_chunks = []
            if manual_context:
                context_chunks.extend(manual_context)
            if auto_context:
                context_chunks.extend(auto_context)

            context_count = len(context_chunks)
            if context_count > 0:
                logger.info(
                    f"[{thread_name}] 📚 Total context: {context_count} chunks (manual: {manual_count}, auto: {auto_count})"
                )

            # Build messages array with conversation history
            messages = self._build_messages_with_history(
                prompt, message_id, context_chunks if context_chunks else None
            )

            # Start periodic "Processing" notifications to Broca so user knows something is happening
            # Notices are sent with increasing delays: immediate, then 10s, 12s, 14.4s, ... up to 20s max
            processing_cancel_event = threading.Event()
            self._start_processing_notices(
                processing_cancel_event, message_id, thread_name
            )

            # Determine model to use - either from config or auto-detect
            logger.info(f"[{thread_name}] STEP 3/6: Determining which model to use...")
            model_to_use = self.config.model_name
            if not model_to_use:
                logger.info(f"[{thread_name}] No model in config, auto-detecting...")
                model_to_use = self.inference.get_running_model()
                if not model_to_use:
                    logger.error(
                        f"[{thread_name}] ✗ Auto-detection failed: no running model found"
                    )
                    raise ValueError(
                        "No model specified in config and no running instances found"
                    )
                logger.info(f"[{thread_name}] ✓ Auto-detected model: {model_to_use}")
            else:
                logger.info(
                    f"[{thread_name}] ✓ Using model from config: {model_to_use}"
                )

            # Call inference backend
            logger.info(f"[{thread_name}] STEP 4/6: Calling AI inference API...")
            logger.info(f"[{thread_name}] API Configuration:")
            logger.info(
                f"[{thread_name}]   Backend: {self.inference.backend_type.value}"
            )
            logger.info(f"[{thread_name}]   Model: {model_to_use}")
            logger.info(f"[{thread_name}]   Temperature: {self.config.temperature}")
            logger.info(
                f"[{thread_name}]   Max tokens: {self.config.max_tokens if self.config.max_tokens else 'default'}"
            )
            logger.info(
                f"[{thread_name}] ⏳ Waiting for AI inference (this may take a while)..."
            )

            api_start_time = datetime.now()
            # Build API call parameters with streaming enabled
            # Use messages array built with conversation history
            api_params = {
                "model": model_to_use,
                "messages": messages,
                "temperature": self.config.temperature,
                "stream": True,  # Enable streaming for faster response
            }
            logger.info(f"[{thread_name}]   Messages in array: {len(messages)}")
            # DEBUG: Log each message in the array to diagnose history issues
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_preview = (
                    content[:100] + "..." if len(content) > 100 else content
                )
                logger.info(
                    f"[{thread_name}]   Message {i}: role={role}, len={len(content)}, preview={content_preview!r}"
                )

            # Validate context window before API call
            fits, context_warning = self.validate_message_context(
                messages, model_to_use
            )
            if context_warning:
                logger.warning(f"[{thread_name}] {context_warning}")
                # Also send warning to persistent log and output topic
                plog = get_persistent_log()
                plog.log("cortex", "warning", context_warning)

                # If context doesn't fit and we have sliding window, this is critical
                if not fits and self._model_capabilities:
                    if self._model_capabilities.has_sliding_window:
                        logger.error(
                            f"[{thread_name}] 🚨 CRITICAL: Context exceeds sliding window! "
                            "The model will NOT be able to see your context/history. "
                            "Response may be incorrect or claim lack of information."
                        )

            # Only add max_tokens if explicitly set
            if self.config.max_tokens is not None:
                api_params["max_tokens"] = self.config.max_tokens

            # Add tools if available and supported
            # Skip tools when doing heavy RAG to prevent model confusion
            context_size = sum(
                len(c.get("content", "")) for c in (context_chunks or [])
            )
            skip_tools_for_rag = context_size > 10000  # 10KB threshold

            if self.available_tools and not skip_tools_for_rag:
                api_params["tools"] = self.available_tools
                tool_names = [t["function"]["name"] for t in self.available_tools]
                logger.info(f"[{thread_name}] 🔧 Tools enabled: {tool_names}")
            elif self.available_tools and skip_tools_for_rag:
                logger.info(
                    f"[{thread_name}] 🔧 Tools DISABLED for this request - "
                    f"RAG context is {context_size:,} chars (threshold: 10,000)"
                )

            logger.info(f"[{thread_name}] 🌊 Starting STREAMING API call...")
            # Access underlying client for raw streaming (complex processing requires OpenAI format)
            stream = self.inference.backend.client.chat.completions.create(**api_params)

            # Process streaming response
            logger.info(f"[{thread_name}] STEP 5/6: Processing streaming response...")
            full_response_text = ""
            streaming_display_text = (
                ""  # Clean text for chat display (excludes thinking blocks)
            )
            word_buffer = ""
            sentence_buffer = (
                ""  # Accumulates complete sentences until word threshold is met
            )
            chunk_count = 0
            chunks_sent = 0
            first_chunk_time = None
            streaming_message_id = None  # ID of the assistant message being streamed
            last_db_update_chunk = 0  # Track when we last updated the DB
            DB_UPDATE_INTERVAL = 10  # Update DB every N chunks

            # Growing word threshold - starts at 5 words, grows by 50% each send
            # This creates a "ramping up" effect for TTS audio generation
            INITIAL_WORD_THRESHOLD = 5
            GROWTH_FACTOR = 1.50  # 50% increase
            current_word_threshold = INITIAL_WORD_THRESHOLD

            # Punctuation marks that indicate sentence boundaries for TTS
            SENTENCE_PUNCTUATION = {".", "!", "?"}  # End of sentence markers
            _PAUSE_PUNCTUATION = {
                ",",
                ";",
                ":",
            }  # Mid-sentence pauses (kept for future use)

            logger.info(f"[{thread_name}] 📦 Growing chunk sizes enabled:")
            logger.info(
                f"[{thread_name}]   Initial word threshold: {INITIAL_WORD_THRESHOLD}"
            )
            logger.info(
                f"[{thread_name}]   Growth factor: {GROWTH_FACTOR} (50% increase per chunk)"
            )
            logger.info(
                f"[{thread_name}]   Sentence boundaries: {SENTENCE_PUNCTUATION}"
            )
            logger.info(
                f"[{thread_name}] 🧠 Tracking thinking blocks in real-time - will skip sending during <think>/<thinking> tags"
            )

            # Tool call tracking for streaming
            accumulated_tool_calls: list[dict] = []
            finish_reason = None

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Track finish_reason (will be "tool_calls" if model wants to call tools)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Accumulate tool calls from streaming chunks
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            # Each tool call chunk has an index
                            tc_index = tc.index if hasattr(tc, "index") else 0

                            # Expand accumulated_tool_calls list if needed
                            while len(accumulated_tool_calls) <= tc_index:
                                accumulated_tool_calls.append(
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            # Accumulate tool call data
                            if hasattr(tc, "id") and tc.id:
                                accumulated_tool_calls[tc_index]["id"] = tc.id
                            if hasattr(tc, "function") and tc.function:
                                if hasattr(tc.function, "name") and tc.function.name:
                                    accumulated_tool_calls[tc_index]["function"][
                                        "name"
                                    ] += tc.function.name
                                if (
                                    hasattr(tc.function, "arguments")
                                    and tc.function.arguments
                                ):
                                    accumulated_tool_calls[tc_index]["function"][
                                        "arguments"
                                    ] += tc.function.arguments

                    if hasattr(delta, "content") and delta.content:
                        chunk_count += 1
                        content = delta.content
                        full_response_text += content

                        if first_chunk_time is None:
                            first_chunk_time = datetime.now()
                            time_to_first_chunk = (
                                first_chunk_time - api_start_time
                            ).total_seconds()
                            logger.info(
                                f"[{thread_name}] ⚡ First chunk received in {time_to_first_chunk:.2f}s"
                            )
                            # Cancel the periodic processing notices now that we have a response
                            processing_cancel_event.set()

                        # Accumulate clean text for display (excluding thinking blocks)
                        if not in_thinking:
                            streaming_display_text += content

                            # Create assistant message on first non-thinking content
                            if (
                                streaming_message_id is None
                                and self.chat_store is not None
                                and streaming_display_text.strip()
                            ):
                                try:
                                    conv_id = (
                                        self.chat_store.get_active_conversation_id()
                                    )
                                    if conv_id:
                                        streaming_message_id = (
                                            self.chat_store.add_assistant_message(
                                                conv_id, streaming_display_text
                                            )
                                        )
                                        logger.info(
                                            f"[{thread_name}] 💬 Created streaming assistant message: {streaming_message_id}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"[{thread_name}] ⚠ Failed to create streaming message: {e}"
                                    )

                            # Periodically update the message in DB
                            elif (
                                streaming_message_id is not None
                                and (chunk_count - last_db_update_chunk)
                                >= DB_UPDATE_INTERVAL
                            ):
                                try:
                                    self.chat_store.update_message(
                                        streaming_message_id, streaming_display_text
                                    )
                                    last_db_update_chunk = chunk_count
                                except Exception as e:
                                    logger.warning(
                                        f"[{thread_name}] ⚠ Failed to update streaming message: {e}"
                                    )

                        # Check for thinking block tags in this chunk
                        if "<think>" in content or "<thinking>" in content:
                            if not in_thinking:
                                logger.info(
                                    f"[{thread_name}] 🧠 Entered thinking block - pausing Kafka sends"
                                )
                                in_thinking = True

                                # Send "Thinking..." message to Broca
                                thinking_message = {
                                    "text": "Thinking, give me a moment...",
                                    "id": f"{message_id}_thinking",
                                    "prompt_id": message_id,
                                    "model": model_to_use,
                                    "is_thinking_notice": True,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                logger.info(
                                    f"[{thread_name}] 💭 Sending 'Thinking...' notice to Broca"
                                )
                                self.producer.send(
                                    self.config.kafka_output_topic,
                                    value=thinking_message,
                                )

                        if "</think>" in content or "</thinking>" in content:
                            if in_thinking:
                                logger.info(
                                    f"[{thread_name}] 🧠 Exited thinking block - resuming Kafka sends"
                                )
                                in_thinking = False
                                # Clear buffers to avoid sending thinking content
                                word_buffer = ""
                                sentence_buffer = ""
                                continue

                        # Only accumulate content if we're not in a thinking block
                        if not in_thinking:
                            word_buffer += content

                            # Check for sentence-ending punctuation in the buffer
                            while True:
                                # Find the earliest sentence-ending punctuation
                                earliest_pos = -1
                                for punct in SENTENCE_PUNCTUATION:
                                    pos = word_buffer.find(punct)
                                    if pos != -1 and (
                                        earliest_pos == -1 or pos < earliest_pos
                                    ):
                                        earliest_pos = pos

                                # If we found sentence punctuation, move complete sentence to sentence_buffer
                                if earliest_pos != -1:
                                    # Extract sentence up to and including the punctuation
                                    complete_sentence = word_buffer[: earliest_pos + 1]
                                    # Keep the rest in word_buffer
                                    word_buffer = word_buffer[earliest_pos + 1 :]

                                    # Add to sentence buffer
                                    sentence_buffer += complete_sentence

                                    # Check if we've accumulated enough words to send
                                    current_word_count = len(sentence_buffer.split())
                                    if current_word_count >= current_word_threshold:
                                        # Strip markdown formatting for TTS
                                        tts_chunk = strip_markdown(sentence_buffer)

                                        # Only send if there's actual content after cleaning
                                        if tts_chunk.strip():
                                            chunks_sent += 1

                                            chunk_output = {
                                                "text": tts_chunk,
                                                "id": f"{message_id}_chunk_{chunks_sent}",
                                                "prompt_id": message_id,
                                                "model": model_to_use,
                                                "is_chunk": True,
                                                "chunk_number": chunks_sent,
                                                "word_threshold": int(
                                                    current_word_threshold
                                                ),
                                                "timestamp": datetime.now().isoformat(),
                                            }

                                            logger.info(
                                                f"[{thread_name}] 📤 Sending chunk #{chunks_sent} ({current_word_count} words >= threshold {int(current_word_threshold)}, {len(tts_chunk)} chars)"
                                            )
                                            self.producer.send(
                                                self.config.kafka_output_topic,
                                                value=chunk_output,
                                            )

                                            # Grow the threshold by 20% for next chunk (ramping up effect)
                                            old_threshold = current_word_threshold
                                            current_word_threshold *= GROWTH_FACTOR
                                            logger.info(
                                                f"[{thread_name}] 📈 Word threshold increased: {int(old_threshold)} → {int(current_word_threshold)} words"
                                            )

                                        # Clear sentence buffer after sending
                                        sentence_buffer = ""
                                else:
                                    # No more sentence punctuation in word_buffer, break and continue accumulating
                                    break

            # Send any remaining content in buffers (only if not in thinking block)
            # Combine sentence_buffer (complete sentences) + word_buffer (incomplete sentence)
            remaining_content = sentence_buffer + word_buffer
            if remaining_content.strip() and not in_thinking:
                tts_chunk = strip_markdown(remaining_content)

                if tts_chunk.strip():
                    chunks_sent += 1
                    final_word_count = len(remaining_content.split())

                    chunk_output = {
                        "text": tts_chunk,
                        "id": f"{message_id}_chunk_{chunks_sent}",
                        "prompt_id": message_id,
                        "model": model_to_use,
                        "is_chunk": True,
                        "chunk_number": chunks_sent,
                        "is_final": True,
                        "word_threshold": int(current_word_threshold),
                        "timestamp": datetime.now().isoformat(),
                    }

                    logger.info(
                        f"[{thread_name}] 📤 Sending FINAL chunk #{chunks_sent} ({final_word_count} words, {len(tts_chunk)} chars)"
                    )
                    self.producer.send(
                        self.config.kafka_output_topic, value=chunk_output
                    )

            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()

            logger.info(f"[{thread_name}] ✓ Streaming completed!")
            logger.info(
                f"[{thread_name}]   Total duration: {api_duration:.2f} seconds ({api_duration / 60:.1f} minutes)"
            )
            logger.info(f"[{thread_name}]   Chunks received: {chunk_count}")
            logger.info(f"[{thread_name}]   Chunks sent to Kafka: {chunks_sent}")
            logger.info(
                f"[{thread_name}]   Full response length: {len(full_response_text)} characters"
            )
            logger.info(f"[{thread_name}]   Finish reason: {finish_reason}")

            # Log AI response to persistent log
            plog = get_persistent_log()
            plog.log(
                component="cortex",
                direction="ai_response",
                content={
                    "full_response": full_response_text,
                    "model": model_to_use,
                    "finish_reason": finish_reason,
                },
                message_id=message_id,
                metadata={
                    "chunk_count": chunk_count,
                    "chunks_sent": chunks_sent,
                    "api_duration": api_duration,
                    "response_length": len(full_response_text),
                },
            )

            # Handle tool calls if the model requested them
            if finish_reason == "tool_calls" and accumulated_tool_calls:
                logger.info(f"[{thread_name}] 🔧 TOOL CALLS DETECTED!")
                logger.info(
                    f"[{thread_name}]   Number of tool calls: {len(accumulated_tool_calls)}"
                )
                for i, tc in enumerate(accumulated_tool_calls):
                    logger.info(
                        f"[{thread_name}]   Tool {i + 1}: {tc['function']['name']}"
                    )

                # Get conversation ID for storing tool messages
                conversation_id = None
                if self.chat_store is not None:
                    conversation_id = self.chat_store.get_active_conversation_id()

                # Execute the tool calls
                tool_results = self._execute_tool_calls(
                    accumulated_tool_calls, conversation_id, message_id
                )

                # Build assistant message with tool_calls for the API
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": accumulated_tool_calls,
                }

                # Extend messages with assistant tool call and results
                messages.append(assistant_tool_msg)
                messages.extend(tool_results)

                # Make a follow-up API call with tool results (non-streaming for simplicity)
                logger.info(
                    f"[{thread_name}] 🔄 Making follow-up API call with tool results..."
                )
                try:
                    # Use inference client for follow-up call
                    follow_up_response = self.inference.complete(
                        messages=messages,
                        model=model_to_use,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=self.available_tools if self.available_tools else None,
                    )
                    follow_up_content = follow_up_response.content or ""

                    logger.info(
                        f"[{thread_name}] ✓ Follow-up response received ({len(follow_up_content)} chars)"
                    )

                    # Add follow-up response to full_response_text for TTS
                    if follow_up_content.strip():
                        full_response_text += "\n" + follow_up_content

                        # Send follow-up response to Broca for TTS
                        chunks_sent += 1
                        follow_up_chunk = {
                            "text": strip_markdown(follow_up_content),
                            "id": f"{message_id}_follow_up",
                            "prompt_id": message_id,
                            "model": model_to_use,
                            "is_chunk": True,
                            "chunk_number": chunks_sent,
                            "is_final": True,
                            "timestamp": datetime.now().isoformat(),
                        }
                        logger.info(
                            f"[{thread_name}] 📤 Sending follow-up response to Broca"
                        )
                        self.producer.send(
                            self.config.kafka_output_topic, value=follow_up_chunk
                        )

                        # Store follow-up response in chat history
                        if self.chat_store is not None and conversation_id:
                            try:
                                self.chat_store.add_assistant_message(
                                    conversation_id, follow_up_content
                                )
                                logger.info(
                                    f"[{thread_name}] 💬 Stored follow-up response in chat history"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[{thread_name}] ⚠ Failed to store follow-up: {e}"
                                )

                except Exception as e:
                    logger.error(f"[{thread_name}] ✗ Follow-up API call failed: {e}")
                    # Send error message to Broca
                    error_msg = f"Tool execution completed but follow-up failed: {e}"
                    self.producer.send(
                        self.config.kafka_output_topic,
                        value={
                            "text": error_msg,
                            "id": f"{message_id}_error",
                            "prompt_id": message_id,
                            "is_error": True,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

            # Extract and log thinking block statistics from full response
            cleaned_text, thinking_blocks = extract_thinking_blocks(full_response_text)

            if thinking_blocks:
                logger.info(f"[{thread_name}] 🧠 THINKING BLOCKS DETECTED")
                stats = compute_thinking_stats(thinking_blocks)
                logger.info(
                    f"[{thread_name}]   Number of thinking blocks: {stats['block_count']}"
                )
                logger.info(
                    f"[{thread_name}]   Total words in thinking: {stats['total_words']}"
                )
                logger.info(
                    f"[{thread_name}]   Total characters in thinking: {stats['total_characters']}"
                )

            logger.info(f"[{thread_name}] STEP 6/6: Cleanup and finalization...")

            # Clean up context from database after successful processing
            if context_count > 0:
                self._cleanup_context(message_id)

            # Final update of assistant response in chat history
            if self.chat_store is not None:
                try:
                    if streaming_message_id is not None:
                        # Update the streaming message with final cleaned text
                        self.chat_store.update_message(
                            streaming_message_id, cleaned_text
                        )
                        logger.info(
                            f"[{thread_name}] 💬 Final update of streaming message ({len(cleaned_text)} chars)"
                        )
                    else:
                        # Fallback: create message if streaming didn't create one
                        conversation_id = self.chat_store.get_active_conversation_id()
                        if conversation_id and cleaned_text.strip():
                            self.chat_store.add_assistant_message(
                                conversation_id, cleaned_text
                            )
                            logger.info(
                                f"[{thread_name}] 💬 Stored assistant response in chat history ({len(cleaned_text)} chars)"
                            )
                except Exception as e:
                    logger.warning(
                        f"[{thread_name}] ⚠ Failed to store assistant response in chat history: {e}"
                    )

            logger.info("▲" * 60)
            logger.info(
                f"[{thread_name}] ✓✓✓ MESSAGE PROCESSING COMPLETED SUCCESSFULLY ✓✓✓"
            )
            logger.info(
                f"[{thread_name}]   Sent {chunks_sent} chunk(s) with growing word threshold ({INITIAL_WORD_THRESHOLD} → {int(current_word_threshold)} words)"
            )
            if context_count > 0:
                logger.info(
                    f"[{thread_name}]   Used {context_count} context chunks from hippocampus"
                )
            if self.chat_store is not None:
                logger.info(
                    f"[{thread_name}]   Chat history: messages stored for conversation continuity"
                )
            logger.info("▲" * 60)

        except Exception as e:
            logger.error("✗" * 60)
            logger.error(f"[{thread_name}] ✗✗✗ ERROR PROCESSING MESSAGE ✗✗✗")
            logger.error("✗" * 60)
            logger.error(f"[{thread_name}] Error type: {type(e).__name__}")
            logger.error(f"[{thread_name}] Error message: {str(e)}")
            logger.error(f"[{thread_name}] Full traceback:", exc_info=True)

            # Send error message to output topic
            logger.info(f"[{thread_name}] Preparing error message for Kafka...")
            error_message = {
                "text": f"Error processing prompt: {str(e)}",
                "id": f"{message_id}_error",
                "prompt_id": message_id,
                "error": True,
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"[{thread_name}] Error message prepared: {error_message}")

            # Log error to persistent log
            plog = get_persistent_log()
            plog.log(
                component="cortex",
                direction="error",
                content={"error": str(e), "error_type": type(e).__name__},
                message_id=message_id,
                metadata={"error_message": error_message},
            )

            logger.info(f"[{thread_name}] Attempting to send error to output topic...")
            try:
                self.producer.send(self.config.kafka_output_topic, value=error_message)
                logger.info(
                    f"[{thread_name}] ✓ Error message sent to Kafka successfully"
                )
            except Exception as send_error:
                logger.error(
                    f"[{thread_name}] ✗ Failed to send error message: {send_error}",
                    exc_info=True,
                )
            logger.error("✗" * 60)

        finally:
            # Ensure processing notices are cancelled
            if processing_cancel_event is not None:
                processing_cancel_event.set()

    def _process_message_wrapper(self, message_value, message_count, message_metadata):
        """Wrapper to process message in a worker thread"""
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        logger.info("=" * 80)
        logger.info(
            f"[THREAD WORKER] Thread {thread_name} (ID: {thread_id}) STARTED processing message #{message_count}"
        )
        logger.info(
            f"[THREAD WORKER] Thread pool status: {len(self.active_futures)} active workers"
        )
        logger.info(
            f"[THREAD WORKER] Message metadata: partition={message_metadata['partition']}, offset={message_metadata['offset']}"
        )
        logger.info(
            f"[THREAD WORKER] Message value: {message_value[:100]}..."
            if len(str(message_value)) > 100
            else f"[THREAD WORKER] Message value: {message_value}"
        )
        logger.info("=" * 80)

        start_time = datetime.now()
        try:
            logger.info(f"[THREAD WORKER {thread_name}] Calling process_message()...")
            self.process_message(message_value)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info("=" * 80)
            logger.info(
                f"[THREAD WORKER {thread_name}] ✓ SUCCESSFULLY processed message #{message_count}"
            )
            logger.info(
                f"[THREAD WORKER {thread_name}] Processing took {duration:.2f} seconds"
            )
            logger.info("=" * 80)
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error("=" * 80)
            logger.error(
                f"[THREAD WORKER {thread_name}] ✗ FAILED to process message #{message_count}"
            )
            logger.error(
                f"[THREAD WORKER {thread_name}] Error after {duration:.2f} seconds: {e}"
            )
            logger.error(
                f"[THREAD WORKER {thread_name}] Error type: {type(e).__name__}"
            )
            logger.error("=" * 80, exc_info=True)

    def _cleanup_completed_futures(self):
        """Remove completed futures from the active list"""
        before_count = len(self.active_futures)
        logger.debug(f"[CLEANUP] Starting cleanup: {before_count} futures in list")
        self.active_futures = [f for f in self.active_futures if not f.done()]
        after_count = len(self.active_futures)
        cleaned = before_count - after_count
        logger.info(
            f"[CLEANUP] Removed {cleaned} completed futures ({before_count} → {after_count})"
        )
        if cleaned > 0:
            logger.info(f"[CLEANUP] Thread pool now has {after_count} active workers")

    def start(self):
        """Start consuming messages"""
        logger.info("=" * 60)
        logger.info("STARTING CONSUMER")
        logger.info("=" * 60)
        logger.info(f"Consumer topic: {self.config.kafka_input_topic}")
        logger.info(f"Consumer group: {self.config.kafka_consumer_group}")
        logger.info(f"Bootstrap servers: {self.config.kafka_bootstrap_servers}")
        logger.info("Using threaded message processing to prevent Kafka rebalancing")
        logger.info("Waiting for prompts...")
        logger.info("=" * 60)

        # Check model capabilities at startup (detects sliding window attention, etc.)
        logger.info("Checking model capabilities...")
        self.check_model_capabilities()
        if self._model_capabilities and self._model_capabilities.has_sliding_window:
            logger.error("=" * 60)
            logger.error(
                "⚠️  WARNING: Model has sliding window attention - RAG may not work!"
            )
            logger.error("=" * 60)

        # Detect tool call support at startup
        logger.info("Checking tool call support...")
        tools_supported = self.check_tool_support()
        logger.info(f"Tool call support: {'YES' if tools_supported else 'NO'}")

        # Discover available AI tools if tool calls are supported
        if tools_supported:
            logger.info("Discovering AI tools...")
            self.available_tools = self.tool_client.discover_tools()
            if self.available_tools:
                tool_names = [t["function"]["name"] for t in self.available_tools]
                logger.info(
                    f"Discovered {len(self.available_tools)} tools: {tool_names}"
                )
            else:
                logger.info("No AI tools available (none configured or healthy)")
        else:
            logger.info("Skipping tool discovery (tool calls not supported)")
        logger.info("=" * 60)

        message_count = 0

        try:
            logger.info("[MAIN THREAD] Entering main consumer loop...")
            logger.info(
                "[MAIN THREAD] Main thread will continuously poll Kafka while workers process messages"
            )
            logger.info(
                "[MAIN THREAD] This prevents Kafka rebalancing during long-running AI inference"
            )

            for message in self.consumer:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    logger.warning(
                        "[MAIN THREAD] Shutdown event detected, stopping consumer loop"
                    )
                    break

                message_count += 1
                receive_timestamp = datetime.now()

                logger.info("\n" + "█" * 80)
                logger.info(
                    f"[MAIN THREAD] ▼▼▼ MESSAGE #{message_count} RECEIVED FROM KAFKA ▼▼▼"
                )
                logger.info("█" * 80)
                logger.info(
                    f"[MAIN THREAD] Receive timestamp: {receive_timestamp.isoformat()}"
                )
                logger.info("[MAIN THREAD] Kafka message details:")
                logger.info(f"[MAIN THREAD]   Topic: {message.topic}")
                logger.info(f"[MAIN THREAD]   Partition: {message.partition}")
                logger.info(f"[MAIN THREAD]   Offset: {message.offset}")
                logger.info(f"[MAIN THREAD]   Timestamp: {message.timestamp}")
                logger.info(f"[MAIN THREAD]   Timestamp type: {message.timestamp_type}")
                logger.info(f"[MAIN THREAD]   Key: {message.key}")
                logger.info(
                    f"[MAIN THREAD]   Value preview: {str(message.value)[:150]}..."
                )
                logger.info(
                    f"[MAIN THREAD]   Value length: {len(str(message.value))} characters"
                )

                # Check if configuration has changed (quick operation, safe to do in main thread)
                logger.debug("[MAIN THREAD] Checking for configuration changes...")
                self._check_config_reload()
                logger.debug("[MAIN THREAD] Configuration check complete")

                # Get thread pool stats before submission
                active_count = len(self.active_futures)
                logger.info("[MAIN THREAD] Thread pool status BEFORE submission:")
                logger.info(f"[MAIN THREAD]   Active workers: {active_count}")
                logger.info("[MAIN THREAD]   Max workers: 3")
                logger.info(f"[MAIN THREAD]   Available capacity: {3 - active_count}")

                # Submit message processing to thread pool instead of blocking
                logger.info(
                    f"[MAIN THREAD] ➤ Submitting message #{message_count} to thread pool..."
                )
                message_metadata = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                }

                submit_timestamp = datetime.now()
                future = self.executor.submit(
                    self._process_message_wrapper,
                    message.value,
                    message_count,
                    message_metadata,
                )
                self.active_futures.append(future)
                submit_complete_timestamp = datetime.now()
                submit_duration = (
                    submit_complete_timestamp - submit_timestamp
                ).total_seconds()

                logger.info(
                    f"[MAIN THREAD] ✓ Message #{message_count} submitted successfully"
                )
                logger.info(
                    f"[MAIN THREAD]   Submission took: {submit_duration * 1000:.2f}ms"
                )
                logger.info(
                    f"[MAIN THREAD]   Active workers now: {len(self.active_futures)}"
                )
                logger.info(
                    "[MAIN THREAD]   Worker thread will process message in background"
                )
                logger.info(
                    "[MAIN THREAD]   Main thread is now FREE to poll Kafka again"
                )
                logger.info("█" * 80)

                # Periodically clean up completed futures to avoid memory growth
                if message_count % 10 == 0:
                    logger.info(
                        "[MAIN THREAD] Periodic cleanup triggered (every 10 messages)"
                    )
                    self._cleanup_completed_futures()
                elif message_count % 5 == 0:
                    logger.info(
                        f"[MAIN THREAD] Thread pool status: {len(self.active_futures)} active workers"
                    )

                logger.info(
                    f"[MAIN THREAD] Ready to receive next message (total processed so far: {message_count})\n"
                )

        except KeyboardInterrupt:
            logger.info("\n" + "🛑" * 60)
            logger.info("[MAIN THREAD] ⚠️  KEYBOARD INTERRUPT RECEIVED (Ctrl+C)")
            logger.info("🛑" * 60)
            logger.info("[MAIN THREAD] Initiating graceful shutdown...")
        except Exception as e:
            logger.error("💥" * 60)
            logger.error("[MAIN THREAD] ✗✗✗ FATAL ERROR IN CONSUMER LOOP ✗✗✗")
            logger.error("💥" * 60)
            logger.error(f"[MAIN THREAD] Error type: {type(e).__name__}")
            logger.error(f"[MAIN THREAD] Error: {e}", exc_info=True)
            raise
        finally:
            logger.info("🔧" * 60)
            logger.info("[SHUTDOWN] CLEANUP AND SHUTDOWN SEQUENCE STARTING")
            logger.info("🔧" * 60)
            logger.info(f"[SHUTDOWN] Total messages received: {message_count}")

            # Signal shutdown to any waiting threads
            logger.info("[SHUTDOWN] Setting shutdown event to stop worker threads...")
            self.shutdown_event.set()
            logger.info("[SHUTDOWN] Shutdown event set")

            # Wait for in-flight message processing to complete
            in_flight_count = len(self.active_futures)
            if in_flight_count > 0:
                logger.info(
                    f"[SHUTDOWN] ⏳ Waiting for {in_flight_count} in-flight message(s) to complete..."
                )
                logger.info("[SHUTDOWN] Each message has 30 seconds timeout")
                for i, future in enumerate(self.active_futures, 1):
                    logger.info(
                        f"[SHUTDOWN] Waiting for message {i}/{in_flight_count}..."
                    )
                    try:
                        future.result(timeout=30)  # Wait up to 30 seconds per message
                        logger.info(
                            f"[SHUTDOWN] ✓ Message {i}/{in_flight_count} completed successfully"
                        )
                    except TimeoutError:
                        logger.error(
                            f"[SHUTDOWN] ✗ Message {i}/{in_flight_count} timed out after 30 seconds"
                        )
                    except Exception as e:
                        logger.error(
                            f"[SHUTDOWN] ✗ Message {i}/{in_flight_count} failed: {e}"
                        )
                logger.info("[SHUTDOWN] ✓ All in-flight messages processed")
            else:
                logger.info("[SHUTDOWN] No in-flight messages to wait for")

            # Shutdown thread pool
            logger.info("[SHUTDOWN] Shutting down thread pool executor...")
            logger.info("[SHUTDOWN]   Max workers: 3")
            logger.info("[SHUTDOWN]   Waiting for all workers to finish...")
            self.executor.shutdown(wait=True, cancel_futures=False)
            logger.info("[SHUTDOWN] ✓ Thread pool executor shut down")

            # Close Kafka connections
            logger.info("[SHUTDOWN] Closing Kafka connections...")
            logger.info("[SHUTDOWN]   1/3: Closing Kafka consumer...")
            self.consumer.close()
            logger.info("[SHUTDOWN]   ✓ Consumer closed")

            logger.info(
                "[SHUTDOWN]   2/3: Flushing Kafka producer (ensuring all messages sent)..."
            )
            self.producer.flush()
            logger.info("[SHUTDOWN]   ✓ Producer flushed")

            logger.info("[SHUTDOWN]   3/3: Closing Kafka producer...")
            self.producer.close()
            logger.info("[SHUTDOWN]   ✓ Producer closed")

            logger.info("✅" * 60)
            logger.info("[SHUTDOWN] ✓✓✓ CONSUMER SHUTDOWN COMPLETE ✓✓✓")
            logger.info("✅" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CONSUMER APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        config = load_config()
        consumer = ExoConsumer(config)
        consumer.start()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR DURING STARTUP")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("CONSUMER APPLICATION EXITING")
    logger.info("=" * 60)
