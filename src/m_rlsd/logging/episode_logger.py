"""Episode logging for complete inference session tracking."""

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """A single event in an episode."""
    event_id: str
    role: str  # "user", "assistant", "tool", "observation"
    turn_id: int
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    """Complete episode log."""
    episode_id: str
    prompt_id: str
    benchmark: str
    question: str
    original_image_path: str
    events: List[Event] = field(default_factory=list)
    final_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    reward: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeLogger:
    """Log complete inference episodes with all events."""

    def __init__(
        self,
        output_dir: str,
        image_subdir: str = "images",
        copy_original_images: bool = True,
    ):
        """Initialize episode logger.
        
        Args:
            output_dir: Base directory for episode logs
            image_subdir: Subdirectory for images relative to output_dir
            copy_original_images: Copy original images to output directory
        """
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / image_subdir
        self.copy_original_images = copy_original_images
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Current episode being logged
        self.current_episode: Optional[EpisodeLog] = None
        self.episode_count = 0

    def start_episode(
        self,
        episode_id: str,
        prompt_id: str,
        benchmark: str,
        question: str,
        image_path: str,
    ) -> None:
        """Start logging a new episode.
        
        Args:
            episode_id: Unique episode identifier
            prompt_id: Original prompt identifier
            benchmark: Benchmark name
            question: The question/instruction
            image_path: Path to original image
        """
        self.current_episode = EpisodeLog(
            episode_id=episode_id,
            prompt_id=prompt_id,
            benchmark=benchmark,
            question=question,
            original_image_path=image_path,
        )
        
        # Copy original image if requested
        if self.copy_original_images and Path(image_path).exists():
            dest_path = self.image_dir / f"{episode_id}_original.png"
            shutil.copy2(image_path, dest_path)
            self.current_episode.original_image_path = str(dest_path)
        
        self.episode_count += 1

    def log_user_turn(self, question: str, image_path: Optional[str] = None) -> None:
        """Log user turn with question and optional image.
        
        Args:
            question: User question
            image_path: Optional path to image
        """
        if not self.current_episode:
            raise RuntimeError("No episode in progress")
        
        segments = [{"type": "text", "content": question}]
        if image_path:
            segments.append({"type": "image", "path": image_path})
        
        event = Event(
            event_id=f"u{len(self.current_episode.events)}",
            role="user",
            turn_id=0,
            segments=segments,
        )
        self.current_episode.events.append(event)

    def log_assistant_turn(
        self,
        reasoning: str,
        code: Optional[str] = None,
        turn_id: int = 0,
    ) -> None:
        """Log assistant turn with reasoning and optional code.
        
        Args:
            reasoning: Reasoning text
            code: Optional Python code
            turn_id: Turn identifier
        """
        if not self.current_episode:
            raise RuntimeError("No episode in progress")
        
        segments = [{"type": "reasoning", "content": reasoning}]
        if code:
            segments.append({"type": "code", "language": "python", "content": code})
        
        event = Event(
            event_id=f"a{len(self.current_episode.events)}",
            role="assistant",
            turn_id=turn_id,
            segments=segments,
        )
        self.current_episode.events.append(event)

    def log_observation(
        self,
        observation_text: str,
        images: Optional[List[str]] = None,
        execution_status: str = "success",
        from_action_event_id: Optional[str] = None,
        turn_id: int = 0,
    ) -> None:
        """Log tool/observation turn.
        
        Args:
            observation_text: Observation output text
            images: Optional list of image paths
            execution_status: Status of execution ("success", "error", "timeout")
            from_action_event_id: Event ID of the action that generated this observation
            turn_id: Turn identifier
        """
        if not self.current_episode:
            raise RuntimeError("No episode in progress")
        
        segments = [{"type": "observation_text", "content": observation_text}]
        
        if images:
            for img_path in images:
                segments.append({"type": "image", "path": img_path})
        
        event = Event(
            event_id=f"o{len(self.current_episode.events)}",
            role="tool",
            turn_id=turn_id,
            segments=segments,
            metadata={
                "execution_status": execution_status,
                "from_action_event_id": from_action_event_id,
            }
        )
        self.current_episode.events.append(event)

    def log_final_answer(self, answer: str) -> None:
        """Log final answer.
        
        Args:
            answer: The final answer text
        """
        if not self.current_episode:
            raise RuntimeError("No episode in progress")
        
        self.current_episode.final_answer = answer

    def finalize_episode(
        self,
        ground_truth_answer: Optional[str] = None,
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Finalize and save episode.
        
        Args:
            ground_truth_answer: Ground truth answer for evaluation
            reward: Computed reward/score
            metadata: Additional metadata
            
        Returns:
            Path where episode was saved
        """
        if not self.current_episode:
            raise RuntimeError("No episode in progress")
        
        episode = self.current_episode
        episode.ground_truth_answer = ground_truth_answer
        episode.reward = reward
        if metadata:
            episode.metadata.update(metadata)
        
        # Save to JSON
        save_path = self.output_dir / f"{episode.episode_id}.json"
        
        # Convert to dict for JSON serialization
        episode_dict = {
            "episode_id": episode.episode_id,
            "prompt_id": episode.prompt_id,
            "benchmark": episode.benchmark,
            "question": episode.question,
            "original_image_path": episode.original_image_path,
            "events": [
                {
                    "event_id": e.event_id,
                    "role": e.role,
                    "turn_id": e.turn_id,
                    "segments": e.segments,
                    "metadata": e.metadata,
                }
                for e in episode.events
            ],
            "final_answer": episode.final_answer,
            "ground_truth_answer": episode.ground_truth_answer,
            "reward": episode.reward,
            "timestamp": episode.timestamp,
            "metadata": episode.metadata,
        }
        
        with open(save_path, "w") as f:
            json.dump(episode_dict, f, indent=2)
        
        # Reset for next episode
        self.current_episode = None
        
        return str(save_path)

    def validate_episode(self, episode_path: str) -> Dict[str, bool]:
        """Validate a saved episode.
        
        Args:
            episode_path: Path to episode JSON file
            
        Returns:
            Dictionary with validation results
        """
        checks = {
            "file_exists": Path(episode_path).exists(),
            "valid_json": False,
            "has_events": False,
            "has_final_answer": False,
            "all_images_exist": True,
        }
        
        try:
            with open(episode_path, "r") as f:
                episode = json.load(f)
            checks["valid_json"] = True
            
            # Check for events
            events = episode.get("events", [])
            checks["has_events"] = len(events) > 0
            
            # Check for final answer
            checks["has_final_answer"] = bool(episode.get("final_answer"))
            
            # Check image paths
            for event in events:
                for segment in event.get("segments", []):
                    if segment.get("type") == "image":
                        img_path = segment.get("path")
                        if img_path and not Path(img_path).exists():
                            checks["all_images_exist"] = False
                            break
                            
        except Exception:
            pass
        
        return checks

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "episodes_logged": self.episode_count,
            "output_directory": str(self.output_dir),
            "image_directory": str(self.image_dir),
            "current_episode_in_progress": self.current_episode is not None,
        }
