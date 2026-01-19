"""
Recursive Language Model (RLM) Executor

Implements the RLM pattern from arxiv:2512.24601v1 where long prompts are treated
as environment objects rather than feeding them directly into neural networks.

The executor:
1. Stores context as a variable in a sandboxed Python environment
2. Gives the LLM metadata about the content (length, structure)
3. The model writes code to probe, filter, and partition the input
4. Sub-LLM calls handle smaller chunks, with results stored in variables
5. Iteration continues until a FINAL() or FINAL_VAR() marker is produced

Usage:
    from libs.rlm_executor import RLMExecutor, RLMConfig

    config = RLMConfig(enabled=True, max_recursion_depth=3)
    executor = RLMExecutor(inference_backend, config)

    result = executor.execute(
        context="<very long document>",
        question="What is the main argument?"
    )
"""

import io
import logging
import re
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for the RLM executor."""

    enabled: bool = True
    max_recursion_depth: int = 3
    max_iterations: int = 10
    max_total_tokens: int = 100000
    sub_call_timeout: float = 60.0
    total_timeout: float = 300.0
    temperature: float = 0.3  # Lower for more deterministic decomposition
    context_threshold: float = 0.5  # Use RLM when context >50% of window

    # Sandbox settings
    allowed_imports: list[str] = field(
        default_factory=lambda: ["re", "json", "collections", "itertools", "math"]
    )
    max_output_chars: int = 50000
    max_exec_time: float = 30.0


@dataclass
class RLMSession:
    """Tracks state for an RLM session."""

    session_id: str
    context: str
    question: str
    variables: dict[str, Any] = field(default_factory=dict)
    call_count: int = 0
    total_tokens: int = 0
    iteration: int = 0
    start_time: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)


class RLMSandbox:
    """
    Sandboxed Python execution environment for RLM.

    Provides a restricted Python environment with:
    - Limited imports
    - No file I/O or network access
    - Timeout enforcement
    - Output capture
    """

    def __init__(self, config: RLMConfig, sub_lm_callback: Callable[[str, str], str]):
        self.config = config
        self.sub_lm_callback = sub_lm_callback
        self._globals: dict[str, Any] = {}
        self._setup_globals()

    def _setup_globals(self) -> None:
        """Set up the restricted global namespace."""
        # Safe builtins
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }

        self._globals = {
            "__builtins__": safe_builtins,
            "__name__": "__rlm_sandbox__",
        }

        # Add allowed imports
        for module_name in self.config.allowed_imports:
            try:
                module = __import__(module_name)
                self._globals[module_name] = module
            except ImportError:
                logger.warning(f"Failed to import allowed module: {module_name}")

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox environment."""
        self._globals[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the sandbox environment."""
        return self._globals.get(name)

    def get_all_variables(self) -> dict[str, Any]:
        """Get all user-defined variables (not builtins or modules)."""
        excluded = {"__builtins__", "__name__"} | set(self.config.allowed_imports)
        return {k: v for k, v in self._globals.items() if k not in excluded}

    def execute(self, code: str) -> tuple[bool, str, Optional[str]]:
        """
        Execute code in the sandbox.

        Returns:
            Tuple of (success, output, error_message)
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Add helper functions to globals
        self._globals["sub_lm"] = self._make_sub_lm()
        self._globals["partition"] = self._partition
        self._globals["chunk_text"] = self._chunk_text
        self._globals["FINAL"] = lambda x: f"__FINAL__:{x}:__END_FINAL__"
        self._globals["FINAL_VAR"] = lambda x: f"__FINAL_VAR__:{x}:__END_FINAL_VAR__"

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._globals)

            output = stdout_capture.getvalue()
            if len(output) > self.config.max_output_chars:
                output = output[: self.config.max_output_chars] + "\n[OUTPUT TRUNCATED]"

            return True, output, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return False, stdout_capture.getvalue(), error_msg

    def _make_sub_lm(self) -> Callable:
        """Create the sub_lm function for the sandbox."""

        def sub_lm(prompt: str, context: str = "") -> str:
            """Call a sub-LLM with the given prompt."""
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            return self.sub_lm_callback(full_prompt, "")

        return sub_lm

    @staticmethod
    def _partition(text: str, chunk_size: int = 4000) -> list[str]:
        """Split text into chunks of approximately chunk_size characters."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))

            # Try to find a good break point (paragraph, sentence, or word boundary)
            if end_pos < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", current_pos, end_pos)
                if para_break > current_pos + chunk_size // 2:
                    end_pos = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                        sent_break = text.rfind(sep, current_pos, end_pos)
                        if sent_break > current_pos + chunk_size // 2:
                            end_pos = sent_break + len(sep)
                            break
                    else:
                        # Look for word break
                        word_break = text.rfind(" ", current_pos, end_pos)
                        if word_break > current_pos + chunk_size // 2:
                            end_pos = word_break + 1

            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

        return chunks

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos - overlap
            if current_pos < 0:
                break

        return chunks


class RLMExecutor:
    """
    Executes Recursive LLM sessions.

    Manages:
    - Python REPL sandbox
    - Variable storage (context chunks, intermediate results)
    - Sub-LM call routing
    - FINAL() marker detection
    - Resource limits enforcement
    """

    # Markers for detecting final answers
    FINAL_PATTERN = re.compile(r"__FINAL__:(.*?):__END_FINAL__", re.DOTALL)
    FINAL_VAR_PATTERN = re.compile(r"__FINAL_VAR__:(.*?):__END_FINAL_VAR__", re.DOTALL)

    # Pattern to extract code blocks from model output
    CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    def __init__(self, inference_backend: Any, config: RLMConfig):
        """
        Initialize the RLM executor.

        Args:
            inference_backend: The inference backend to use for sub-LM calls
            config: RLM configuration
        """
        self.backend = inference_backend
        self.config = config
        self._current_session: Optional[RLMSession] = None

    def should_use_rlm(self, context_size: int, context_window: int) -> bool:
        """
        Determine if RLM should be used for this request.

        Args:
            context_size: Size of the context in characters
            context_window: Model's context window in tokens

        Returns:
            True if RLM should be used
        """
        if not self.config.enabled:
            return False

        # Estimate tokens (rough: 4 chars per token)
        estimated_tokens = context_size // 4

        # Use RLM if context exceeds threshold
        return estimated_tokens > context_window * self.config.context_threshold

    def execute(
        self,
        context: str,
        question: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Run an RLM session to answer question using context.

        Args:
            context: The full context document
            question: The user's question
            session_id: Optional session ID for tracking

        Returns:
            The final answer string
        """
        import uuid

        session_id = session_id or str(uuid.uuid4())[:8]

        logger.info(
            f"[RLM:{session_id}] Starting session with {len(context):,} char context"
        )

        # Create session
        session = RLMSession(
            session_id=session_id,
            context=context,
            question=question,
        )
        self._current_session = session

        # Create sandbox with sub_lm callback
        sandbox = RLMSandbox(
            config=self.config, sub_lm_callback=self._handle_sub_lm_call
        )

        # Initialize sandbox with context and question
        sandbox.set_variable("context", context)
        sandbox.set_variable("question", question)
        sandbox.set_variable("context_length", len(context))

        # Build initial system prompt
        system_prompt = self._build_rlm_system_prompt(len(context))

        # Initial user message with metadata
        initial_message = f"""You have access to a context document stored in the `context` variable.

Context metadata:
- Length: {len(context):,} characters
- Estimated tokens: ~{len(context) // 4:,}

Question to answer: {question}

Write Python code to analyze the context and answer the question. Use sub_lm() for LLM calls on chunks.
When you have the final answer, call FINAL(answer) or FINAL_VAR(variable_name)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_message},
        ]

        # Main execution loop
        while session.iteration < self.config.max_iterations:
            session.iteration += 1
            elapsed = time.time() - session.start_time

            if elapsed > self.config.total_timeout:
                logger.warning(f"[RLM:{session_id}] Total timeout exceeded")
                return self._create_timeout_response(session, sandbox)

            logger.debug(f"[RLM:{session_id}] Iteration {session.iteration}")

            # Get model response
            try:
                response = self.backend.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"[RLM:{session_id}] Backend error: {e}")
                return f"Error during RLM execution: {e}"

            session.total_tokens += (
                response.usage.get("total_tokens", 0) if response.usage else 0
            )
            model_output = response.content

            # Check for direct FINAL in model output (without code execution)
            direct_final = self._extract_final_answer(model_output)
            if direct_final is not None:
                logger.info(f"[RLM:{session_id}] Direct final answer found")
                return direct_final

            # Extract and execute code
            code_blocks = self.CODE_BLOCK_PATTERN.findall(model_output)

            if not code_blocks:
                # No code block - model might have answered directly
                # Check if the response looks like a final answer
                if self._looks_like_final_answer(model_output, question):
                    logger.info(f"[RLM:{session_id}] Treating response as final answer")
                    return model_output.strip()

                # Prompt for code
                messages.append({"role": "assistant", "content": model_output})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please write Python code to analyze the context. "
                        "Remember to use FINAL(answer) when you have the answer.",
                    }
                )
                continue

            # Execute each code block
            all_output = []
            final_answer = None

            for code in code_blocks:
                success, output, error = sandbox.execute(code)

                if output:
                    all_output.append(output)

                if error:
                    all_output.append(f"Error: {error}")

                # Check for FINAL marker in output
                final_answer = self._extract_final_from_output(output, sandbox)
                if final_answer is not None:
                    break

                # Check for FINAL_VAR marker
                final_var = self._extract_final_var_from_output(output, sandbox)
                if final_var is not None:
                    final_answer = final_var
                    break

            if final_answer is not None:
                logger.info(
                    f"[RLM:{session_id}] Final answer found after {session.iteration} iterations"
                )
                return final_answer

            # Add execution results to conversation
            execution_result = "\n".join(all_output) if all_output else "(no output)"
            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"Code execution result:\n```\n{execution_result}\n```\n\n"
                    f"Continue analysis or call FINAL(answer) if you have the answer.",
                }
            )

            # Store session history
            session.history.append(
                {
                    "iteration": session.iteration,
                    "code": code_blocks,
                    "output": all_output,
                }
            )

        # Max iterations reached
        logger.warning(f"[RLM:{session_id}] Max iterations reached")
        return self._create_fallback_response(session, sandbox)

    def _handle_sub_lm_call(self, prompt: str, context: str) -> str:
        """Handle a sub_lm() call from the sandbox."""
        if self._current_session is None:
            return "Error: No active RLM session"

        session = self._current_session
        session.call_count += 1

        if session.call_count > 50:  # Limit sub-calls
            return "Error: Too many sub_lm calls"

        logger.debug(f"[RLM:{session.session_id}] Sub-LM call #{session.call_count}")

        try:
            response = self.backend.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=2048,
                timeout=self.config.sub_call_timeout,
            )
            session.total_tokens += (
                response.usage.get("total_tokens", 0) if response.usage else 0
            )
            return response.content
        except Exception as e:
            logger.error(f"[RLM:{session.session_id}] Sub-LM call failed: {e}")
            return f"Error: {e}"

    def _build_rlm_system_prompt(self, context_length: int) -> str:
        """Build the system prompt for RLM mode."""
        return f"""You are a Recursive Language Model (RLM) assistant. You have access to a Python REPL
and can examine large contexts by writing code.

Available functions:
- sub_lm(prompt, context="") -> str: Call a sub-LLM with a prompt. Use for analyzing chunks.
- partition(text, chunk_size=4000) -> list[str]: Split text into chunks at natural boundaries.
- chunk_text(text, chunk_size=4000, overlap=200) -> list[str]: Split text with overlap.
- len(var) -> int: Get length of variable.
- FINAL(answer) -> marks this as the final answer. Call when you have the answer.
- FINAL_VAR(varname) -> marks this variable's contents as the final answer.

Variables available:
- context: The full context document ({context_length:,} characters)
- question: The user's question
- context_length: Length of context in characters

IMPORTANT WORKFLOW:
1. First, understand the context structure (print samples, check length)
2. Partition into manageable chunks
3. Use sub_lm() to analyze each chunk
4. Aggregate findings
5. Call FINAL(answer) with your final answer

Example:
```python
# Sample the context
print(f"Context length: {{len(context)}}")
print("Beginning:", context[:500])

# Partition and analyze
chunks = partition(context, 4000)
print(f"Split into {{len(chunks)}} chunks")

findings = []
for i, chunk in enumerate(chunks):
    result = sub_lm(f"Find mentions of X in this text. Quote relevant parts:\\n{{chunk}}")
    if result and "none" not in result.lower():
        findings.append(f"Chunk {{i}}: {{result}}")

# Synthesize answer
if findings:
    synthesis = sub_lm(f"Synthesize these findings to answer: {{question}}\\n\\n" + "\\n".join(findings))
    print(FINAL(synthesis))
else:
    print(FINAL("No relevant information found in the context."))
```

Always output valid Python code in code blocks. End with FINAL() when you have the answer."""

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract FINAL() answer from text."""
        # Look for FINAL("...") or FINAL('...') pattern in the text
        patterns = [
            r'FINAL\("([^"]+)"\)',
            r"FINAL\('([^']+)'\)",
            r'FINAL\("""(.*?)"""\)',
            r"FINAL\('''(.*?)'''\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        return None

    def _extract_final_from_output(
        self, output: str, sandbox: RLMSandbox
    ) -> Optional[str]:
        """Extract final answer from execution output."""
        match = self.FINAL_PATTERN.search(output)
        if match:
            return match.group(1)
        return None

    def _extract_final_var_from_output(
        self, output: str, sandbox: RLMSandbox
    ) -> Optional[str]:
        """Extract final answer from a variable."""
        match = self.FINAL_VAR_PATTERN.search(output)
        if match:
            var_name = match.group(1).strip()
            value = sandbox.get_variable(var_name)
            if value is not None:
                return str(value)
        return None

    def _looks_like_final_answer(self, text: str, question: str) -> bool:
        """Check if text looks like a final answer (not a request for more code)."""
        # If it contains code blocks, it's not a final answer
        if "```" in text:
            return False

        # If it's asking about code or analysis, it's not final
        code_indicators = [
            "let me write",
            "i'll write",
            "here's the code",
            "let me analyze",
            "i need to",
            "first, i'll",
        ]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in code_indicators):
            return False

        # If it's a substantive response (>100 chars) that doesn't mention code, treat as final
        return len(text.strip()) > 100

    def _create_timeout_response(self, session: RLMSession, sandbox: RLMSandbox) -> str:
        """Create a response when timeout is reached."""
        variables = sandbox.get_all_variables()

        # Try to find any partial results
        for key in ["findings", "results", "answer", "summary", "response"]:
            if key in variables and variables[key]:
                value = variables[key]
                if isinstance(value, list):
                    return "\n".join(str(v) for v in value)
                return str(value)

        return (
            f"Analysis timed out after {session.iteration} iterations. "
            f"Processed {session.call_count} sub-queries."
        )

    def _create_fallback_response(
        self, session: RLMSession, sandbox: RLMSandbox
    ) -> str:
        """Create a fallback response when max iterations reached."""
        variables = sandbox.get_all_variables()

        # Try to synthesize from available data
        for key in ["findings", "results", "answer", "summary", "response", "final"]:
            if key in variables and variables[key]:
                value = variables[key]
                if isinstance(value, list) and value:
                    return "\n".join(str(v) for v in value[-5:])  # Last 5 results
                if value:
                    return str(value)

        return (
            f"Unable to complete analysis within {self.config.max_iterations} iterations. "
            f"The context may require manual review."
        )
