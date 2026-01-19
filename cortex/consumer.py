#!/usr/bin/env python3
# consumer.py
"""
Cortex Consumer - AI Inference Pipeline Component

Consumes prompts from Kafka, processes them through the AI inference backend,
and sends responses to the output topic for TTS processing.
"""

from kafka import KafkaConsumer, KafkaProducer
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

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
from libs.text_processing import extract_thinking_blocks, compute_thinking_stats
from libs.streaming_processor import StreamingProcessor, ToolCallAccumulator
from libs.kafka_messages import KafkaMessageFactory
from libs.context_formatter import ContextFormatter

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

        model_name_env = self.cfg.get("MODEL_NAME", "").strip()
        self.model_name = model_name_env if model_name_env else None

        self.temperature = self.cfg.get_float("TEMPERATURE", 0.7)

        max_tokens_env = self.cfg.get("MAX_TOKENS", "").strip()
        self.max_tokens = int(max_tokens_env) if max_tokens_env else None

        self.log_level = self.cfg.get("LOG_LEVEL", "INFO")

        self.context_db_path = self.cfg.get_path(
            "CONTEXT_DB_PATH", "./hippocampus/data/context.db"
        )
        self.cleanup_context_after_use = self.cfg.get_bool(
            "CLEANUP_CONTEXT_AFTER_USE", True
        )

        self.chat_db_path = self.cfg.get_path("CHAT_DB_PATH", "./cortex/data/chat.db")
        self.chat_history_enabled = self.cfg.get_bool("CHAT_HISTORY_ENABLED", True)

        reset_patterns_str = self.cfg.get(
            "CHAT_RESET_PATTERNS",
            "/reset,reset conversation,new conversation,start over",
        )
        self.chat_reset_patterns = [
            p.strip().lower() for p in reset_patterns_str.split(",")
        ]

        self.system_prompt = self.cfg.get(
            "CORTEX_SYSTEM_PROMPT",
            "You are an AI assistant who provides help to a user based on provided "
            "context and instruction prompts. Your goal is to answer questions and "
            "complete tasks based on the user's input.",
        )

        # Tool result context limit as percentage of effective context window (0.0-1.0)
        self.tool_result_context_limit = self.cfg.get_float(
            "CORTEX_TOOL_RESULT_CONTEXT_LIMIT", 0.30
        )

    def reload(self) -> bool:
        """Check for config changes and reload if needed"""
        if self.cfg.reload():
            self._load()
            return True
        return False


def load_config() -> CortexConfig:
    """Load configuration from environment variables"""
    logger.info("Loading configuration...")
    cortex_cfg = CortexConfig(config)

    import logging

    logger.setLevel(getattr(logging, cortex_cfg.log_level.upper(), logging.INFO))

    logger.info(f"  Kafka: {cortex_cfg.kafka_bootstrap_servers}")
    logger.info(f"  Input: {cortex_cfg.kafka_input_topic}")
    logger.info(f"  Output: {cortex_cfg.kafka_output_topic}")
    logger.info(
        f"  Model: {cortex_cfg.model_name if cortex_cfg.model_name else 'auto-detect'}"
    )

    return cortex_cfg


class ExoConsumer:
    def __init__(self, config: CortexConfig):
        logger.info("Initializing ExoConsumer...")
        self.config = config
        self.msg_factory = KafkaMessageFactory()
        self.context_formatter = ContextFormatter(config.system_prompt)

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            config.kafka_input_topic,
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_deserializer=lambda m: m.decode("utf-8"),
            auto_offset_reset=config.kafka_auto_offset_reset,
            enable_auto_commit=True,
            group_id=config.kafka_consumer_group,
            max_poll_interval_ms=600000,
            session_timeout_ms=60000,
            heartbeat_interval_ms=10000,
        )

        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        # Initialize inference client
        self.inference = get_inference_client()
        logger.info(f"Inference backend: {self.inference.backend_type.value}")

        self._model_capabilities = None
        self._init_stores()

        # Thread pool for message processing
        self.executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="msg-processor"
        )
        self.active_futures = []
        self.shutdown_event = threading.Event()

        # Tool support
        self._tools_supported: bool | None = None
        self._state = get_state()
        self.tool_client = ToolClient(self.config.cfg)
        self.available_tools: list[dict] = []

        logger.info("ExoConsumer initialized")

    def _init_stores(self):
        """Initialize context and chat stores"""
        try:
            self.context_store = ContextStore(self.config.context_db_path)
            logger.info(f"Context store: {self.config.context_db_path}")
        except Exception as e:
            logger.warning(f"Context store failed: {e}")
            self.context_store = None

        if self.config.chat_history_enabled:
            try:
                self.chat_store = ChatStore(self.config.chat_db_path)
                logger.info(f"Chat store: {self.config.chat_db_path}")
            except Exception as e:
                logger.warning(f"Chat store failed: {e}")
                self.chat_store = None
        else:
            self.chat_store = None

    def check_tool_support(self, force_redetect: bool = False) -> bool:
        """Check if tool calls are supported by the current model."""
        if self._tools_supported is not None and not force_redetect:
            return self._tools_supported

        self._tools_supported = self.inference.supports_tools()

        try:
            model_name = self.inference.get_running_model() or "unknown"
            self._state.set_bool("cortex.tools_supported", self._tools_supported)
            self._state.set_string("cortex.tools_model", model_name)
            self._state.set_string(
                "cortex.tools_checked_at", datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"Failed to store tool support in state: {e}")

        return self._tools_supported

    def check_model_capabilities(self, model: str | None = None) -> None:
        """Check and cache model capabilities."""
        capabilities = self.inference.get_model_capabilities(model)
        if capabilities is None:
            logger.warning("Could not retrieve model capabilities")
            return

        self._model_capabilities = capabilities

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
            logger.warning(f"Failed to store capabilities in state: {e}")

        logger.info(f"Model: {capabilities.model_id}")
        logger.info(f"  Context: {capabilities.context_length:,} tokens")

        if capabilities.has_sliding_window:
            logger.error(
                f"  SLIDING WINDOW: {capabilities.sliding_window:,} tokens - RAG may not work!"
            )

    def validate_message_context(
        self, messages: list[dict], model: str | None = None
    ) -> tuple[bool, str | None]:
        """Validate that messages fit within the model's context window."""
        model_to_check = model or self.inference.get_running_model()

        cache_stale = (
            self._model_capabilities is None
            or self._model_capabilities.model_id != model_to_check
        )
        if cache_stale:
            self.check_model_capabilities(model_to_check)

        if self._model_capabilities is None:
            return (True, None)

        return self._model_capabilities.check_context_fit(messages)

    def _get_manual_context_documents(self) -> list[dict]:
        """Get manually selected context documents."""
        if not self.config.context_db_path:
            return []

        try:
            import sqlite3

            conn = sqlite3.connect(self.config.context_db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='context_documents'"
            )
            if cursor.fetchone() is None:
                conn.close()
                return []

            cursor = conn.execute(
                "SELECT id, text, source, added_at FROM context_documents ORDER BY added_at ASC"
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "id": row["id"],
                    "content": row["text"],
                    "source": row["source"],
                    "is_manual": True,
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to get manual context: {e}")
            return []

    def _start_processing_notices(
        self, cancel_event: threading.Event, message_id: str
    ) -> threading.Thread:
        """Start background thread for periodic processing notices."""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Starting processing notices for {message_id}")

        def notice_sender():
            notice_count = 0
            delay = 10.0
            max_delay = 20.0
            growth_factor = 1.2

            try:
                logger.info(
                    f"[notice_sender] Sending initial processing notice for {message_id}"
                )
                notice_count += 1
                msg = self.msg_factory.processing_notice(message_id, notice_count)
                logger.debug(f"[notice_sender] Message: {msg}")
                self.producer.send(self.config.kafka_output_topic, value=msg)
                self.producer.flush()
                logger.info("[notice_sender] Initial notice sent and flushed")

                while not cancel_event.is_set():
                    if cancel_event.wait(timeout=delay):
                        logger.info("[notice_sender] Cancel event received, stopping")
                        return

                    notice_count += 1
                    logger.info(f"[notice_sender] Sending notice #{notice_count}")
                    msg = self.msg_factory.processing_notice(
                        message_id, notice_count, "Processing, please wait..."
                    )
                    self.producer.send(self.config.kafka_output_topic, value=msg)
                    self.producer.flush()

                    delay = min(delay * growth_factor, max_delay)
            except Exception as e:
                logger.error(f"Processing notice error: {e}", exc_info=True)

        thread = threading.Thread(target=notice_sender, daemon=True)
        thread.start()
        # Small yield to ensure the thread has a chance to run and send the initial notice
        # before the main thread starts the API call
        time.sleep(0.1)
        return thread

    def _is_reset_command(self, prompt: str) -> bool:
        """Check if the prompt is a conversation reset command."""
        prompt_lower = prompt.strip().lower()
        for pattern in self.config.chat_reset_patterns:
            if prompt_lower == pattern or prompt_lower.startswith(pattern + " "):
                return True
        return False

    def _handle_reset_command(self, message_id: str) -> bool:
        """Handle a conversation reset command."""
        if self.chat_store is None:
            return False

        try:
            new_conv_id = self.chat_store.reset_conversation()
            msg = self.msg_factory.reset_confirmation(message_id, new_conv_id)
            self.producer.send(self.config.kafka_output_topic, value=msg)
            logger.info(f"Conversation reset. New ID: {new_conv_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset conversation: {e}")
            return False

    def _build_messages_with_history(
        self, prompt: str, message_id: str, context_chunks: list[dict] | None = None
    ) -> list[dict]:
        """Build messages array with conversation history."""
        thread_name = threading.current_thread().name
        messages = []

        skip_tools = self.context_formatter.should_skip_tools_for_rag(context_chunks)

        tool_system_prompt = None
        if not skip_tools:
            tool_system_prompt = self.context_formatter.build_tool_system_prompt(
                self.available_tools
            )
            if tool_system_prompt:
                messages.append({"role": "system", "content": tool_system_prompt})

        if self.chat_store is None:
            if context_chunks:
                combined = self.context_formatter.format_context_with_prompt(
                    context_chunks, prompt
                )
                messages.append({"role": "user", "content": combined})
            else:
                messages.append({"role": "user", "content": prompt})
            return messages

        try:
            conversation_id = self.chat_store.get_or_create_active_conversation()
            history = self.chat_store.get_conversation_messages(conversation_id)
            already_injected_ids = self.chat_store.get_conversation_context_ids(
                conversation_id
            )

            for msg in history:
                message_type = msg.get("message_type", "message")
                if message_type not in ("context_user", "context_ack"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

            new_context_chunks = None
            if context_chunks:
                new_context_chunks = [
                    ctx
                    for ctx in context_chunks
                    if ctx.get("id") not in already_injected_ids
                ]

            if new_context_chunks:
                combined_content = self.context_formatter.format_context_with_prompt(
                    new_context_chunks, prompt
                )
                messages.append({"role": "user", "content": combined_content})

                context_metadata = {
                    "context_ids": [
                        ctx.get("id") for ctx in new_context_chunks if ctx.get("id")
                    ],
                    "sources": [
                        ctx.get("source")
                        for ctx in new_context_chunks
                        if ctx.get("source")
                    ],
                    "chunk_count": len(new_context_chunks),
                    "original_prompt": prompt,
                }
                self.chat_store.add_user_message(
                    conversation_id,
                    combined_content,
                    prompt_id=message_id,
                    message_type="context_with_prompt",
                    metadata=context_metadata,
                )

                contexts_to_track = []
                for ctx in new_context_chunks:
                    ctx_id = ctx.get("id")
                    content = ctx.get("content", "")
                    if ctx_id:
                        content_hash = hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest()
                        contexts_to_track.append((ctx_id, content_hash))
                if contexts_to_track:
                    self.chat_store.add_conversation_contexts_batch(
                        conversation_id, contexts_to_track
                    )
            else:
                messages.append({"role": "user", "content": prompt})
                self.chat_store.add_user_message(
                    conversation_id, prompt, prompt_id=message_id
                )

            logger.info(f"[{thread_name}] Built {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"[{thread_name}] Error building history: {e}")
            messages = []
            if tool_system_prompt:
                messages.append({"role": "system", "content": tool_system_prompt})
            if context_chunks:
                combined = self.context_formatter.format_context_with_prompt(
                    context_chunks, prompt
                )
                messages.append({"role": "user", "content": combined})
            else:
                messages.append({"role": "user", "content": prompt})
            return messages

    def _cleanup_context(self, prompt_id: str):
        """Clean up context from database after use"""
        if self.context_store is None or not self.config.cleanup_context_after_use:
            return

        try:
            deleted = self.context_store.delete_contexts_for_prompt(prompt_id)
            logger.debug(f"Cleaned up {deleted} context chunks")
        except Exception as e:
            logger.warning(f"Failed to clean up context: {e}")

    def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        conversation_id: str | None,
        prompt_id: str | None,
    ) -> list[dict]:
        """Execute tool calls and return results."""
        thread_name = threading.current_thread().name
        results = []

        if self.chat_store is not None and conversation_id:
            try:
                tool_calls_data = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": tc.get("function", {}),
                    }
                    for tc in tool_calls
                ]
                self.chat_store.add_tool_call_message(
                    conversation_id, tool_calls_data, prompt_id
                )
            except Exception as e:
                logger.warning(f"[{thread_name}] Failed to store tool call: {e}")

        for tc in tool_calls:
            tool_call_id = tc.get("id", "")
            function_info = tc.get("function", {})
            function_name = function_info.get("name", "")
            arguments_str = function_info.get("arguments", "{}")

            logger.info(f"[{thread_name}] Executing tool: {function_name}")

            try:
                arguments = json.loads(arguments_str)
                result = self.tool_client.call_tool(function_name, arguments)

                if result.get("success"):
                    result_content = result.get("result", "")

                    # Apply context limiting for search tool results
                    if function_name == "search" and isinstance(result_content, dict):
                        max_tokens = self._get_tool_result_context_limit()
                        query = arguments.get("query", "")
                        results_list = result_content.get("results", [])
                        total_chars = sum(len(r.get("text", "")) for r in results_list)
                        logger.info(
                            f"[{thread_name}] Search returned {len(results_list)} results, "
                            f"~{total_chars} chars, limit={max_tokens} tokens (~{max_tokens * 4} chars)"
                        )

                        trimmed_result, overflow, orig_count = (
                            self._trim_search_results(result_content, max_tokens)
                        )

                        if overflow:
                            # Even the first result is too large - notify user
                            self._send_context_overflow_notice(query, prompt_id or "")
                            result_content = json.dumps(
                                {
                                    "error": "Context overflow",
                                    "message": "Search results exceed context window limit",
                                    "query": query,
                                }
                            )
                        else:
                            result_content = trimmed_result
                            returned_count = len(trimmed_result.get("results", []))
                            logger.info(
                                f"[{thread_name}] After trimming: {returned_count}/{orig_count} results"
                            )
                    elif function_name == "search":
                        logger.warning(
                            f"[{thread_name}] Search result_content is not a dict: {type(result_content)}"
                        )

                    # Ensure result_content is a string (tools may return dicts)
                    if not isinstance(result_content, str):
                        result_content = json.dumps(result_content)
                else:
                    error = result.get("error", {})
                    result_content = f"Error: {error.get('type', 'Unknown')}: {error.get('message', 'Unknown error')}"

            except json.JSONDecodeError as e:
                result_content = f"Error: Invalid arguments JSON: {e}"
            except Exception as e:
                result_content = f"Error: {type(e).__name__}: {e}"

            if self.chat_store is not None and conversation_id:
                try:
                    self.chat_store.add_tool_result_message(
                        conversation_id, tool_call_id, function_name, result_content
                    )
                except Exception as e:
                    logger.warning(f"[{thread_name}] Failed to store tool result: {e}")

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )

        return results

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text. Rough estimate: ~4 chars per token."""
        return len(text) // 4

    def _get_tool_result_context_limit(self) -> int:
        """Get the maximum tokens allowed for tool results."""
        limit_pct = self.config.tool_result_context_limit

        if self._model_capabilities is None:
            # Default to a conservative 8K tokens if capabilities unknown
            return int(8000 * limit_pct / 0.30)

        # Use effective context (considers sliding window if present)
        effective_ctx = self._model_capabilities.effective_context
        return int(effective_ctx * limit_pct)

    def _trim_search_results(
        self, result_data: dict, max_tokens: int
    ) -> tuple[dict, bool, int]:
        """
        Trim search results to fit within token limit.

        Args:
            result_data: The search result dict with 'results' list
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (trimmed_result_data, had_overflow, original_count)
        """
        if not isinstance(result_data, dict):
            return result_data, False, 0

        results = result_data.get("results", [])
        if not results:
            return result_data, False, 0

        original_count = len(results)
        trimmed_results = []
        total_tokens = 0

        # Results are already sorted by relevance (distance) from ChromaDB
        for result in results:
            text = result.get("text", "")
            result_tokens = self._estimate_tokens(text)

            # Check if this single result exceeds the limit
            if not trimmed_results and result_tokens > max_tokens:
                # Even the first/most relevant result is too large
                return result_data, True, original_count

            if total_tokens + result_tokens <= max_tokens:
                trimmed_results.append(result)
                total_tokens += result_tokens
            else:
                # Stop adding results once we exceed the limit
                break

        # Update the result data with trimmed results
        trimmed_data = result_data.copy()
        trimmed_data["results"] = trimmed_results
        trimmed_data["results_trimmed"] = len(trimmed_results) < original_count
        trimmed_data["original_result_count"] = original_count
        trimmed_data["returned_result_count"] = len(trimmed_results)

        return trimmed_data, False, original_count

    def _send_context_overflow_notice(self, query: str, message_id: str):
        """Send a context overflow notification to Broca."""
        thread_name = threading.current_thread().name
        logger.warning(
            f"[{thread_name}] Search result context overflow for query: {query[:50]}..."
        )

        overflow_msg = self.msg_factory.context_overflow_notice(
            message_id=f"{message_id}_overflow",
            query=query,
        )
        self.producer.send(self.config.kafka_output_topic, value=overflow_msg)
        self.producer.flush()

    def _check_config_reload(self):
        """Check if config has changed and reload if needed"""
        old_model_name = self.config.model_name

        if self.config.reload():
            logger.info("Configuration reloaded")

            if self.inference.reload():
                logger.info("Inference client reloaded")

            if self.config.model_name != old_model_name:
                logger.info("Re-detecting tool support after model change...")
                self._tools_supported = None
                tools_supported = self.check_tool_support(force_redetect=True)
                if tools_supported:
                    self.available_tools = self.tool_client.discover_tools()
                else:
                    self.available_tools = []

    def process_message(self, message):
        """Process a single prompt message and send to AI backend"""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Processing message...")

        processing_cancel_event = None
        stream_processor = StreamingProcessor(
            initial_word_threshold=5,
            growth_factor=1.5,
        )
        tool_accumulator = ToolCallAccumulator()

        try:
            # Parse message
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                    prompt = parsed.get("text", "") or parsed.get("prompt", "")
                    message_id = parsed.get(
                        "id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    )
                    context_available = parsed.get("context_available", False)
                    contexts_stored = parsed.get("contexts_stored", 0)
                except json.JSONDecodeError:
                    prompt = message
                    message_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    context_available = False
                    contexts_stored = 0
            else:
                prompt = message.get("text", "") or message.get("prompt", "")
                message_id = message.get(
                    "id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
                context_available = message.get("context_available", False)
                contexts_stored = message.get("contexts_stored", 0)

            if not prompt:
                logger.warning(f"[{thread_name}] Empty prompt, skipping")
                return

            logger.info(f"[{thread_name}] Prompt: {prompt[:100]}...")

            # Log to persistent log
            plog = get_persistent_log()
            plog.log(
                component="cortex",
                direction="received",
                content={"prompt": prompt},
                message_id=message_id,
                topic=self.config.kafka_input_topic,
            )

            # Handle reset command
            if self._is_reset_command(prompt):
                if self._handle_reset_command(message_id):
                    return

            # Gather context
            manual_context = self._get_manual_context_documents()
            auto_context = []
            if context_available and contexts_stored > 0 and self.context_store:
                auto_context = self.context_store.get_contexts_for_prompt(message_id)

            context_chunks = manual_context + auto_context
            context_count = len(context_chunks)

            # Build messages
            messages = self._build_messages_with_history(
                prompt, message_id, context_chunks if context_chunks else None
            )

            # Start processing notices
            processing_cancel_event = threading.Event()
            self._start_processing_notices(processing_cancel_event, message_id)

            # Determine model
            model_to_use = self.config.model_name or self.inference.get_running_model()
            if not model_to_use:
                raise ValueError("No model available")

            logger.info(f"[{thread_name}] Using model: {model_to_use}")

            # Validate context
            fits, context_warning = self.validate_message_context(
                messages, model_to_use
            )
            if context_warning:
                logger.warning(f"[{thread_name}] {context_warning}")

            # Build API params
            api_params = {
                "model": model_to_use,
                "messages": messages,
                "temperature": self.config.temperature,
                "stream": True,
            }

            if self.config.max_tokens:
                api_params["max_tokens"] = self.config.max_tokens

            # Check if we should include tools
            # Re-verify tool support if model changed since last check
            use_tools = False
            skip_tools = self.context_formatter.should_skip_tools_for_rag(
                context_chunks
            )
            if self.available_tools and not skip_tools:
                cached_model = self._state.get_string("cortex.tools_model")
                if cached_model != model_to_use:
                    # Model changed, re-check tool support
                    logger.info(
                        f"[{thread_name}] Model changed ({cached_model} -> {model_to_use}), "
                        "re-checking tool support"
                    )
                    self._tools_supported = None
                    tools_supported = self.check_tool_support(force_redetect=True)
                    if tools_supported:
                        use_tools = True
                    else:
                        logger.info(
                            f"[{thread_name}] Model {model_to_use} does not support tools"
                        )
                        self.available_tools = []
                elif self._tools_supported:
                    use_tools = True

            if use_tools:
                api_params["tools"] = self.available_tools

            # Make streaming API call
            api_start_time = datetime.now()
            try:
                stream = self.inference.backend.client.chat.completions.create(
                    **api_params
                )
            except Exception as e:
                # Check if error is due to tool support
                error_msg = str(e).lower()
                if "does not support tools" in error_msg or "tools" in error_msg:
                    logger.warning(
                        f"[{thread_name}] Model rejected tools, retrying without: {e}"
                    )
                    # Mark tools as unsupported and retry
                    self._tools_supported = False
                    self.available_tools = []
                    api_params.pop("tools", None)
                    stream = self.inference.backend.client.chat.completions.create(
                        **api_params
                    )
                else:
                    raise

            # Process stream
            first_chunk_time = None
            finish_reason = None
            streaming_message_id = None
            last_db_update_chunk = 0
            DB_UPDATE_INTERVAL = 10
            was_in_thinking = False

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Accumulate tool calls
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_accumulator.process_delta(delta.tool_calls)

                    if hasattr(delta, "content") and delta.content:
                        content = delta.content

                        if first_chunk_time is None:
                            first_chunk_time = datetime.now()
                            ttfc = (first_chunk_time - api_start_time).total_seconds()
                            logger.info(
                                f"[{thread_name}] First chunk in {ttfc:.2f}s, cancelling processing notices"
                            )
                            processing_cancel_event.set()

                        # Check for thinking block entry
                        was_in_thinking = stream_processor.in_thinking

                        # Process content
                        tts_chunks = stream_processor.process_content(content)

                        # Send thinking notice if just entered thinking
                        if stream_processor.in_thinking and not was_in_thinking:
                            msg = self.msg_factory.thinking_notice(
                                message_id, model_to_use
                            )
                            self.producer.send(
                                self.config.kafka_output_topic, value=msg
                            )

                        # Create streaming message in DB
                        if (
                            streaming_message_id is None
                            and self.chat_store
                            and stream_processor.display_text.strip()
                            and not stream_processor.in_thinking
                        ):
                            conv_id = self.chat_store.get_active_conversation_id()
                            if conv_id:
                                streaming_message_id = (
                                    self.chat_store.add_assistant_message(
                                        conv_id, stream_processor.display_text
                                    )
                                )

                        # Periodically update DB
                        if (
                            streaming_message_id
                            and (
                                stream_processor.state.chunk_count
                                - last_db_update_chunk
                            )
                            >= DB_UPDATE_INTERVAL
                        ):
                            try:
                                self.chat_store.update_message(
                                    streaming_message_id, stream_processor.display_text
                                )
                                last_db_update_chunk = (
                                    stream_processor.state.chunk_count
                                )
                            except Exception:
                                pass

                        # Send TTS chunks
                        for tts_chunk in tts_chunks:
                            msg = self.msg_factory.tts_chunk(
                                text=tts_chunk.text,
                                message_id=f"{message_id}_chunk_{tts_chunk.chunk_number}",
                                prompt_id=message_id,
                                chunk_number=tts_chunk.chunk_number,
                                model=model_to_use,
                                word_threshold=tts_chunk.word_threshold,
                                is_final=tts_chunk.is_final,
                            )
                            self.producer.send(
                                self.config.kafka_output_topic, value=msg
                            )

            # Flush remaining content
            final_chunk = stream_processor.flush()
            if final_chunk:
                msg = self.msg_factory.tts_chunk(
                    text=final_chunk.text,
                    message_id=f"{message_id}_chunk_{final_chunk.chunk_number}",
                    prompt_id=message_id,
                    chunk_number=final_chunk.chunk_number,
                    model=model_to_use,
                    word_threshold=final_chunk.word_threshold,
                    is_final=True,
                )
                self.producer.send(self.config.kafka_output_topic, value=msg)

            # Flush producer to ensure TTS chunks reach Broca
            self.producer.flush()

            api_duration = (datetime.now() - api_start_time).total_seconds()
            logger.info(
                f"[{thread_name}] Streaming completed in {api_duration:.1f}s, "
                f"{stream_processor.chunks_sent} chunks sent"
            )

            # Log response
            plog.log(
                component="cortex",
                direction="ai_response",
                content={
                    "full_response": stream_processor.full_response,
                    "model": model_to_use,
                    "finish_reason": finish_reason,
                },
                message_id=message_id,
            )

            # Handle tool calls
            if finish_reason == "tool_calls" and tool_accumulator.has_tool_calls():
                logger.info(f"[{thread_name}] Processing tool calls...")

                conversation_id = None
                if self.chat_store:
                    conversation_id = self.chat_store.get_active_conversation_id()

                tool_results = self._execute_tool_calls(
                    tool_accumulator.get_tool_calls(), conversation_id, message_id
                )

                # Build follow-up messages
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_accumulator.get_tool_calls(),
                }
                messages.append(assistant_tool_msg)
                messages.extend(tool_results)

                # Log tool result size
                for tr in tool_results:
                    content_len = len(tr.get("content", ""))
                    logger.info(
                        f"[{thread_name}] Tool result content length: {content_len} chars"
                    )

                try:
                    logger.info(
                        f"[{thread_name}] Sending follow-up with {len(messages)} messages"
                    )
                    follow_up = self.inference.complete(
                        messages=messages,
                        model=model_to_use,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=self.available_tools if self.available_tools else None,
                    )
                    follow_up_content = follow_up.content or ""
                    logger.info(
                        f"[{thread_name}] Follow-up response: {len(follow_up_content)} chars, "
                        f"preview: {follow_up_content[:200] if follow_up_content else '(empty)'}..."
                    )

                    if follow_up_content.strip():
                        from libs.text_processing import strip_markdown

                        chunks_sent = stream_processor.chunks_sent + 1
                        msg = self.msg_factory.follow_up_response(
                            text=strip_markdown(follow_up_content),
                            message_id=f"{message_id}_follow_up",
                            prompt_id=message_id,
                            chunk_number=chunks_sent,
                            model=model_to_use,
                        )
                        self.producer.send(self.config.kafka_output_topic, value=msg)
                        self.producer.flush()

                        if self.chat_store and conversation_id:
                            self.chat_store.add_assistant_message(
                                conversation_id, follow_up_content
                            )
                    else:
                        logger.warning(f"[{thread_name}] Follow-up content was empty!")
                except Exception as e:
                    logger.error(
                        f"[{thread_name}] Follow-up failed: {e}", exc_info=True
                    )

            # Extract thinking stats
            cleaned_text, thinking_blocks = extract_thinking_blocks(
                stream_processor.full_response
            )
            if thinking_blocks:
                stats = compute_thinking_stats(thinking_blocks)
                logger.info(
                    f"[{thread_name}] Thinking: {stats['total_words']} words in {stats['block_count']} block(s)"
                )

            # Cleanup
            if context_count > 0:
                self._cleanup_context(message_id)

            # Final DB update
            if self.chat_store and streaming_message_id:
                self.chat_store.update_message(streaming_message_id, cleaned_text)

            logger.info(f"[{thread_name}] Message processing completed")

        except Exception as e:
            logger.error(f"[{thread_name}] Error: {e}", exc_info=True)

            error_msg = self.msg_factory.error_message(
                message_id=f"{message_id}_error",
                prompt_id=message_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            try:
                self.producer.send(self.config.kafka_output_topic, value=error_msg)
            except Exception:
                pass

            plog = get_persistent_log()
            plog.log(
                component="cortex",
                direction="error",
                content={"error": str(e)},
                message_id=message_id,
            )

        finally:
            if processing_cancel_event:
                processing_cancel_event.set()

    def _process_message_wrapper(self, message_value, message_count, message_metadata):
        """Wrapper to process message in a worker thread"""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Processing message #{message_count}")

        start_time = datetime.now()
        try:
            self.process_message(message_value)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{thread_name}] Completed in {duration:.1f}s")
        except Exception as e:
            logger.error(f"[{thread_name}] Failed: {e}", exc_info=True)

    def _cleanup_completed_futures(self):
        """Remove completed futures from the active list"""
        before = len(self.active_futures)
        self.active_futures = [f for f in self.active_futures if not f.done()]
        cleaned = before - len(self.active_futures)
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} completed futures")

    def start(self):
        """Start consuming messages"""
        logger.info("=" * 60)
        logger.info("STARTING CONSUMER")
        logger.info(f"  Topic: {self.config.kafka_input_topic}")
        logger.info(f"  Group: {self.config.kafka_consumer_group}")
        logger.info("=" * 60)

        self.check_model_capabilities()
        if self._model_capabilities and self._model_capabilities.has_sliding_window:
            logger.error("WARNING: Model has sliding window - RAG may not work!")

        tools_supported = self.check_tool_support()
        logger.info(f"Tool support: {'YES' if tools_supported else 'NO'}")

        if tools_supported:
            self.available_tools = self.tool_client.discover_tools()
            if self.available_tools:
                tool_names = [t["function"]["name"] for t in self.available_tools]
                logger.info(f"Tools: {tool_names}")

        logger.info("Waiting for prompts...")

        message_count = 0

        try:
            for message in self.consumer:
                if self.shutdown_event.is_set():
                    break

                message_count += 1
                logger.info(f"Received message #{message_count}")

                self._check_config_reload()

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

                if message_count % 10 == 0:
                    self._cleanup_completed_futures()

        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Shutting down...")
            self.shutdown_event.set()

            in_flight = len(self.active_futures)
            if in_flight > 0:
                logger.info(f"Waiting for {in_flight} in-flight message(s)...")
                for future in self.active_futures:
                    try:
                        future.result(timeout=30)
                    except Exception:
                        pass

            self.executor.shutdown(wait=True, cancel_futures=False)
            self.consumer.close()
            self.producer.flush()
            self.producer.close()

            logger.info("Consumer shutdown complete")


def _send_startup_error_to_broca(error_msg: str) -> None:
    """Send a startup error message to Broca for TTS notification."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        factory = KafkaMessageFactory()
        msg = factory.error_message(
            message_id="startup_error",
            prompt_id="startup",
            error=error_msg,
            error_type="StartupError",
        )
        output_topic = config.get("CORTEX_OUTPUT_TOPIC", "ai_out")
        producer.send(output_topic, value=msg)
        producer.flush(timeout=5)
        producer.close()
        logger.info(f"Sent startup error to Broca: {error_msg}")
    except Exception as e:
        logger.warning(f"Failed to send startup error to Broca: {e}")


if __name__ == "__main__":
    logger.info("Starting Cortex consumer...")

    try:
        config = load_config()
        consumer = ExoConsumer(config)
        consumer.start()
    except ValueError as e:
        # Check for API key error specifically
        error_str = str(e)
        if "ANTHROPIC_API_KEY" in error_str:
            logger.error(f"API key error: {e}")
            _send_startup_error_to_broca(
                "Anthropic API key not found. Please add ANTHROPIC_API_KEY to your .env file."
            )
        else:
            logger.error(f"Configuration error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise
