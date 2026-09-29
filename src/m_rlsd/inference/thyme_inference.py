"""Thyme-style multimodal inference engine with code execution."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m_rlsd.sandbox.executor import SandboxExecutor, ExecutionResult
from m_rlsd.data.benchmark_loader import BenchmarkExample
from m_rlsd.inference.remote_vlm_client import RemoteVLMClient, RemoteVLMConfig


@dataclass
class Turn:
    """A single turn in a conversation."""
    turn_id: int
    role: str  # "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Complete episode with all turns and final answer."""
    episode_id: str
    prompt_id: str
    benchmark: str
    question: str
    original_image_path: str
    turns: List[Turn] = field(default_factory=list)
    final_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThymeInferenceEngine:
    """Inference engine for Thyme-style multimodal reasoning."""
    
    # System prompt for the agent
    SYSTEM_PROMPT = """You are a helpful AI assistant tasked with solving problems through reasoning and computation.

When given a problem:
1. First, understand what's being asked
2. If you need to perform calculations or analyze data, write Python code in a markdown code block
3. Execute the code mentally or provide the reasoning
4. Extract and state your final answer clearly

For code blocks, use the format:
```python
# Your code here
```

Always be clear about your reasoning and provide a final answer."""

    def __init__(
        self,
        model_path: str,
        backend: str = "local_checkpoint",
        device: str = "cuda",
        max_turns: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        sandbox_enabled: bool = True,
        sandbox_timeout: int = 5,
        use_system_prompt: bool = True,
        raise_on_error: bool = False,
        remote_base_url: Optional[str] = None,
        remote_api_key: str = "EMPTY",
        remote_model: Optional[str] = None,
        remote_timeout_sec: float = 120.0,
        remote_max_retries: int = 2,
        remote_retry_backoff_sec: float = 2.0,
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to LLM model
            backend: Inference backend (`local_checkpoint` or `remote_api`)
            device: Device to use ("cuda", "cpu")
            max_turns: Maximum turns per episode
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            sandbox_enabled: Enable Python code execution
            sandbox_timeout: Timeout for code execution
            use_system_prompt: Whether to use system prompt
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sandbox_enabled = sandbox_enabled
        self.use_system_prompt = use_system_prompt
        self.raise_on_error = raise_on_error
        self.remote_model = remote_model or model_path
        self.model = None
        self.tokenizer = None
        self.remote_client: Optional[RemoteVLMClient] = None

        if self.backend == "local_checkpoint":
            # Initialize model and tokenizer
            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self.model.eval()

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.backend == "remote_api":
            self.remote_client = RemoteVLMClient(
                RemoteVLMConfig(
                    base_url=remote_base_url or "",
                    model=self.remote_model,
                    api_key=remote_api_key,
                    timeout_sec=remote_timeout_sec,
                    max_retries=remote_max_retries,
                    retry_backoff_sec=remote_retry_backoff_sec,
                )
            )
            print(
                "Using remote OpenAI-compatible VLM endpoint "
                f"{self.remote_client.base_url} model={self.remote_model}"
            )
        else:
            raise ValueError(
                "backend must be one of: 'local_checkpoint', 'remote_api'"
            )
        
        # Initialize sandbox if enabled
        self.sandbox = SandboxExecutor(timeout=sandbox_timeout) if sandbox_enabled else None
        
        # Execution statistics
        self.stats = {
            "episodes_generated": 0,
            "total_turns": 0,
            "code_executions": 0,
            "execution_successes": 0,
            "execution_failures": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def infer_episode(
        self,
        example: BenchmarkExample,
        episode_id: str,
    ) -> Episode:
        """Run complete episode for a single example.
        
        Args:
            example: BenchmarkExample with question and image
            episode_id: Unique episode identifier
            
        Returns:
            Complete Episode with all turns
        """
        episode = Episode(
            episode_id=episode_id,
            prompt_id=example.example_id,
            benchmark=example.benchmark,
            question=example.question,
            original_image_path=example.image_path,
            ground_truth_answer=example.ground_truth_answer,
        )
        
        # Add user turn with question
        self._add_user_turn(episode, example.question, example.image_path)
        
        # Multi-turn reasoning loop
        for turn_idx in range(self.max_turns):
            try:
                # Generate assistant response
                assistant_response = self._generate_response(episode)
                
                # Parse response for code or answer
                if self._has_code_block(assistant_response):
                    # Extract and execute code
                    code = self._extract_code(assistant_response)
                    self._add_assistant_turn(episode, assistant_response)
                    
                    if self.sandbox_enabled:
                        execution_result = self.sandbox.execute(code)
                        self._add_observation_turn(episode, execution_result)
                        
                        self.stats["code_executions"] += 1
                        if execution_result.success:
                            self.stats["execution_successes"] += 1
                        else:
                            self.stats["execution_failures"] += 1
                    else:
                        # Just log code without execution
                        self._add_observation_turn(episode, None)
                else:
                    # Final answer
                    self._add_assistant_turn(episode, assistant_response)
                    episode.final_answer = self._extract_answer(assistant_response)
                    break
            except Exception as e:
                episode.metadata["error"] = str(e)
                episode.metadata["error_turn"] = turn_idx
                print(f"Error in turn {turn_idx}: {e}")
                if self.raise_on_error:
                    raise
                break
        
        self.stats["episodes_generated"] += 1
        self.stats["total_turns"] += len(episode.turns)
        
        return episode

    def _add_user_turn(self, episode: Episode, question: str, image_path: str) -> None:
        """Add user turn with question and image."""
        turn = Turn(
            turn_id=len(episode.turns),
            role="user",
            content=f"Question: {question}\nImage: {image_path}",
            metadata={
                "question": question,
                "image_path": image_path,
            }
        )
        episode.turns.append(turn)

    def _add_assistant_turn(self, episode: Episode, response: str) -> None:
        """Add assistant turn with response."""
        turn = Turn(
            turn_id=len(episode.turns),
            role="assistant",
            content=response,
            metadata={"has_code": self._has_code_block(response)}
        )
        episode.turns.append(turn)

    def _add_observation_turn(
        self,
        episode: Episode,
        result: Optional[ExecutionResult],
    ) -> None:
        """Add observation turn from code execution."""
        if result:
            content = f"Execution Status: {'Success' if result.success else 'Failed'}\n"
            if result.stdout:
                content += f"Output:\n{result.stdout}\n"
            if result.stderr:
                content += f"Errors:\n{result.stderr}\n"
            if result.error_message:
                content += f"Error: {result.error_message}\n"
        else:
            content = "Code execution skipped (sandbox disabled)"
        
        turn = Turn(
            turn_id=len(episode.turns),
            role="tool",
            content=content,
            metadata={"execution_success": result.success if result else None}
        )
        episode.turns.append(turn)

    def _generate_response(self, episode: Episode) -> str:
        """Generate next assistant response using model.
        
        Args:
            episode: Current episode
            
        Returns:
            Generated response text
        """
        if self.backend == "remote_api":
            return self._generate_remote_response(episode)

        # Build context from episode turns
        context = self._build_context(episode)
        
        # Tokenize
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()

    def _generate_remote_response(self, episode: Episode) -> str:
        """Generate the next assistant response via remote chat completions."""
        if self.remote_client is None:
            raise RuntimeError("remote client is not initialized")

        messages = self._build_remote_messages(episode)
        response_text, raw_response = self.remote_client.generate_text(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

        usage = raw_response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int):
            self.stats["total_tokens"] += prompt_tokens
        if isinstance(completion_tokens, int):
            self.stats["total_tokens"] += completion_tokens

        return response_text

    def _build_context(self, episode: Episode) -> str:
        """Build conversation context from turns."""
        context_parts = []
        
        # Add system prompt if enabled
        if self.use_system_prompt:
            context_parts.append(f"System: {self.SYSTEM_PROMPT}")
        
        # Add conversation history
        for turn in episode.turns:
            if turn.role == "user":
                context_parts.append(f"User: {turn.content}")
            elif turn.role == "assistant":
                context_parts.append(f"Assistant: {turn.content}")
            elif turn.role == "tool":
                context_parts.append(f"Tool Output:\n{turn.content}")
        
        # Prompt for next response
        context_parts.append("Assistant:")
        return "\n\n".join(context_parts)

    def _build_remote_messages(self, episode: Episode) -> List[Dict[str, Any]]:
        """Build OpenAI-compatible message payload for remote inference."""
        messages: List[Dict[str, Any]] = []

        if self.use_system_prompt:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        first_user_turn = True
        for turn in episode.turns:
            if turn.role == "user":
                image_path = turn.metadata.get("image_path") if turn.metadata else None
                if first_user_turn and image_path:
                    messages.append(
                        RemoteVLMClient.build_multimodal_user_message(
                            turn.content,
                            image_path=image_path,
                        )
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": turn.content,
                        }
                    )
                first_user_turn = False
            elif turn.role == "assistant":
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.content,
                    }
                )
            elif turn.role == "tool":
                # Keep the payload widely compatible by folding tool output into user text.
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool Output:\n{turn.content}",
                    }
                )

        return messages

    def _has_code_block(self, text: str) -> bool:
        """Check if text contains Python code block."""
        return "```python" in text or "```" in text

    def _extract_code(self, text: str) -> str:
        """Extract Python code from response."""
        # Try markdown code blocks first
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try triple backticks without language
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no code blocks found, return empty string
        return ""

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from response."""
        # Look for answer indicators
        patterns = [
            r"(?:final\s+)?answer\s*(?:is|=|:)\s*([^\n]+)",
            r"the\s+answer\s+(?:is|=)\s*([^\n]+)",
            r"result\s*(?:is|=|:)\s*([^\n]+)",
            r"answer:\s*([^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Remove common punctuation
                answer = answer.rstrip('.,!?;:')
                return answer
        
        # Fall back to last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            last_line = lines[-1].rstrip('.,!?;:')
            return last_line
        
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.stats.copy()
