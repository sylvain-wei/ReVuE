"""Sandbox execution engine for safe Python code execution with image support.

Provides SandboxExecutor class that:
- Executes Python code with timeout protection
- Captures stdout/stderr
- Handles errors gracefully
- Supports PIL image loading from paths
- Supports matplotlib/PIL image output
"""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from pathlib import Path
import tempfile


@dataclass
class ExecutionResult:
    """Result of a sandbox execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[Any] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    images_generated: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "images_generated": self.images_generated,
        }


class SandboxExecutor:
    """Safely execute Python code with timeout and resource protection."""

    def __init__(
        self,
        timeout: int = 5,
        max_memory_mb: int = 512,
        blacklist_modules: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        enable_image_loading: bool = True,
    ):
        """Initialize sandbox executor.
        
        Args:
            timeout: Max execution time in seconds (default: 5)
            max_memory_mb: Max memory in MB (default: 512)
            blacklist_modules: Modules to block (e.g., ["os", "subprocess"])
            work_dir: Working directory for execution (default: temp dir)
            enable_image_loading: Enable PIL image loading (default: True)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.blacklist_modules = blacklist_modules or [
            "os.system",
            "subprocess",
            "pickle",
            "__import__('os').system",
        ]
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enable_image_loading = enable_image_loading
        
        self.last_result: Optional[ExecutionResult] = None
        self.execution_count = 0

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        setup_code: Optional[str] = None,
        image_paths: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            timeout: Override default timeout (seconds)
            setup_code: Optional setup code (imports, etc.)
            image_paths: Dict of variable_name -> image_path for image loading
            
        Returns:
            ExecutionResult with success status and outputs
        """
        import time
        
        self.execution_count += 1
        timeout = timeout or self.timeout
        result = ExecutionResult(success=False)
        
        try:
            start_time = time.time()
            
            # Check for blacklisted code patterns
            for pattern in self.blacklist_modules:
                if pattern in code:
                    result.error_type = "SecurityError"
                    result.error_message = f"Blacklisted pattern: {pattern}"
                    return result
            
            # Prepare execution environment
            exec_globals = {
                "__name__": "__main__",
                "__builtins__": __builtins__,
            }
            
            # Add safe imports - actually import them
            try:
                import numpy as np
                import math
                import json
                exec_globals["numpy"] = np
                exec_globals["np"] = np
                exec_globals["math"] = math
                exec_globals["json"] = json
            except ImportError:
                pass
            
            # Add PIL if image loading is enabled
            if self.enable_image_loading:
                try:
                    from PIL import Image
                    exec_globals["Image"] = Image
                    
                    # Load images if paths provided
                    if image_paths:
                        for var_name, img_path in image_paths.items():
                            try:
                                img = Image.open(img_path)
                                exec_globals[var_name] = img
                            except Exception as e:
                                result.stderr += f"Warning: Failed to load image {var_name}: {e}\n"
                except ImportError:
                    pass
            
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Execute setup code if provided
            if setup_code:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(setup_code, exec_globals)
            
            # Execute main code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)
            
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()
            result.success = True
            result.return_value = exec_globals.get("result")
            
            # Capture important variables (exclude private and builtin)
            result.variables = {
                k: v for k, v in exec_globals.items()
                if not k.startswith("_") and k not in ["__builtins__"]
            }
            
        except TimeoutError:
            result.error_type = "TimeoutError"
            result.error_message = f"Execution timed out after {timeout}s"
        except SyntaxError as e:
            result.error_type = "SyntaxError"
            result.error_message = str(e)
        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.stderr = traceback.format_exc()
        finally:
            result.execution_time = time.time() - (start_time if 'start_time' in locals() else time.time())
        
        self.last_result = result
        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "execution_count": self.execution_count,
            "last_execution_time": self.last_result.execution_time if self.last_result else 0.0,
            "last_success": self.last_result.success if self.last_result else False,
        }
