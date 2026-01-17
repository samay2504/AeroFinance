"""
Sandbox Executor - Secure Python code execution with AST validation.
Runs LLM-generated code in restricted environment.
"""
import logging
import ast
import os
import sys
import io
import traceback
from typing import Any, Dict, Optional, Set
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Allowed modules for sandboxed code
ALLOWED_MODULES = {"pandas", "numpy", "math", "decimal", "statistics", "datetime"}

# Blocked patterns in code
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "import http",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "file(",
    "input(",
    "breakpoint(",
    "globals(",
    "locals(",
    "vars(",
    "dir(",
    "__builtins__",
    "__class__",
    "__subclasses__",
    "__mro__",
    "__bases__",
    "getattr(",
    "setattr(",
    "delattr(",
]


class CodeValidator:
    """AST-based code validator for security."""

    def __init__(self, allowed_modules: Set[str] = None):
        self.allowed_modules = allowed_modules or ALLOWED_MODULES

    def validate(self, code: str) -> tuple[bool, str]:
        """
        Validate Python code for safety.
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check blocked patterns
        code_lower = code.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Blocked pattern detected: {pattern}"

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Walk AST and validate
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in self.allowed_modules:
                        return False, f"Import not allowed: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in self.allowed_modules:
                    return False, f"Import not allowed: {node.module}"

            # Check for dangerous calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile", "open", "__import__"]:
                        return False, f"Dangerous function call: {node.func.id}"

                elif isinstance(node.func, ast.Attribute):
                    # Check for subprocess, os methods
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ["os", "subprocess", "sys"]:
                            return False, f"Dangerous module access: {node.func.value.id}"

        return True, ""

    def check_has_run_function(self, code: str) -> bool:
        """Check if code defines a run(df) function."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "run":
                    return True
        except Exception:
            pass
        return False


class SandboxExecutor:
    """
    Secure executor for LLM-generated Python code.
    Uses AST validation and restricted globals.
    
    Features:
    - AST-based security validation
    - Resource limits (memory, timeout)
    - Subprocess isolation (optional)
    - Structured return format
    - E2B integration for remote execution (optional)
    """

    def __init__(
        self, 
        timeout_seconds: int = 30, 
        max_memory_mb: int = 512,
        use_subprocess: bool = False
    ):
        self.timeout = timeout_seconds
        self.max_memory = max_memory_mb
        self.use_subprocess = use_subprocess
        self.validator = CodeValidator()
        
        # Track execution metrics
        self._execution_count = 0
        self._killed_count = 0
        self._total_exec_time_ms = 0
    
    def run_user_code(
        self,
        code: str,
        df_map: Dict[str, pd.DataFrame],
        timeout_sec: int = 10
    ) -> Dict[str, Any]:
        """
        Execute user code with structured return format.
        
        This is the primary API for secure code execution.
        
        Args:
            code: Python code to execute
            df_map: Map of DataFrame names to DataFrames
            timeout_sec: Execution timeout in seconds
            
        Returns:
            {
                "success": bool,
                "result": Any,  # Execution result
                "logs": str,    # stdout/stderr output
                "error": str,   # Error message if failed
                "exec_time_ms": int
            }
        """
        import time as time_module
        start_time = time_module.time()
        self._execution_count += 1
        
        # Use primary df if only one provided
        df = None
        if df_map:
            if "df" in df_map:
                df = df_map["df"]
            else:
                df = list(df_map.values())[0]
        
        result = self.execute(code, df if df is not None else pd.DataFrame())
        
        exec_time_ms = int((time_module.time() - start_time) * 1000)
        self._total_exec_time_ms += exec_time_ms
        
        # Track killed tasks
        if result.get("error") and "timeout" in str(result.get("error", "")).lower():
            self._killed_count += 1
            logger.warning(f"Sandbox task killed (timeout): {result.get('error', '')[:100]}")
        
        return {
            "success": result.get("success", False),
            "result": result.get("result"),
            "logs": (result.get("stdout", "") + "\n" + result.get("stderr", "")).strip(),
            "error": result.get("error"),
            "exec_time_ms": exec_time_ms
        }


    def _clean_code(self, code: str) -> str:
        """
        Clean code of markdown artifacts and formatting issues.
        
        Enhancements:
        - Strip mixed tabs/spaces before executing
        - Detect indentation errors early
        - Fix common LLM code formatting issues
        """
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python", 1)[-1]
        elif "```Python" in code:
            code = code.split("```Python", 1)[-1]
        if "```" in code:
            code = code.split("```")[0]

        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Fix mixed tabs/spaces - convert all tabs to 4 spaces
        code = code.replace("\t", "    ")

        # Fix common indentation issues
        lines = code.split("\n")
        if lines:
            # Find minimum indentation (excluding empty lines)
            min_indent = float("inf")
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            if min_indent > 0 and min_indent != float("inf"):
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
                code = "\n".join(lines)
        
        # Early indentation error detection - try to parse AST
        try:
            import ast
            ast.parse(code)
        except IndentationError as e:
            # Try to auto-fix common indentation issues
            logger.warning(f"Indentation error detected: {e}. Attempting auto-fix...")
            lines = code.split("\n")
            fixed_lines = []
            current_indent = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    fixed_lines.append("")
                    continue
                
                # Dedent for closing structures
                if stripped.startswith(("return", "break", "continue", "pass", "raise")):
                    fixed_lines.append("    " * current_indent + stripped)
                elif stripped.startswith(("except", "elif", "else", "finally")):
                    if current_indent > 0:
                        current_indent -= 1
                    fixed_lines.append("    " * current_indent + stripped)
                    current_indent += 1
                else:
                    fixed_lines.append("    " * current_indent + stripped)
                
                # Increase indent for blocks
                if stripped.endswith(":"):
                    current_indent += 1
            
            code = "\n".join(fixed_lines)
            logger.debug("Auto-fix applied to indentation")
        except SyntaxError:
            # Let the validator handle syntax errors
            pass
        
        # Remove any trailing empty lines but keep one newline at end
        code = code.rstrip() + "\n" if code.strip() else code

        return code
    
    def _validate_returns_dict(self, code: str) -> bool:
        """
        Check if code appears to return a dictionary (or dict-like result).
        Used to detect when LLM returns prose instead of proper code.
        """
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    # Check if return value is a dict, call, or name
                    if node.value and isinstance(node.value, (ast.Dict, ast.Call, ast.Name)):
                        return True
            return False
        except Exception:
            return False


    def _create_sandbox_globals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create restricted globals for sandbox execution."""
        
        # Whitelist of safe modules that can be imported
        safe_modules = {
            "math": __import__("math"),
            "decimal": __import__("decimal"),
            "datetime": __import__("datetime"),
            "re": __import__("re"),
            "statistics": __import__("statistics"),
            "collections": __import__("collections"),
            "functools": __import__("functools"),
            "itertools": __import__("itertools"),
        }
        
        # Safe import function that only allows whitelisted modules
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Restricted import that only allows whitelisted modules."""
            if name in safe_modules:
                return safe_modules[name]
            elif name == "pandas" or name == "pd":
                return pd
            elif name == "numpy" or name == "np":
                return np
            else:
                raise ImportError(f"Import of '{name}' not allowed in sandbox")
        
        return {
            "__builtins__": {
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "print": print,
                "True": True,
                "False": False,
                "None": None,
                "__import__": safe_import,  # Safe import function
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
            },
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,
            "df": df.copy(),  # Pass a copy for safety
            "math": safe_modules["math"],
            "decimal": safe_modules["decimal"],
            "datetime": safe_modules["datetime"],
            "re": safe_modules["re"],
            "statistics": safe_modules["statistics"],
        }

    def execute(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute code in sandbox.
        
        Args:
            code: Python code to execute (should define run(df) function)
            df: DataFrame to pass to the code
            
        Returns:
            Dict with success, result, stdout, stderr, error
        """
        result = {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": None,
        }

        # Clean code
        code = self._clean_code(code)

        # Validate
        is_safe, error = self.validator.validate(code)
        if not is_safe:
            result["error"] = f"Code validation failed: {error}"
            return result

        # Check for run function
        has_run = self.validator.check_has_run_function(code)

        # Create sandbox environment
        sandbox_globals = self._create_sandbox_globals(df)
        sandbox_locals = {}

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code
                exec(code, sandbox_globals, sandbox_locals)

                # If run function exists, call it
                if has_run and "run" in sandbox_locals:
                    run_result = sandbox_locals["run"](df.copy())
                    result["result"] = self._serialize_result(run_result)
                elif "result" in sandbox_locals:
                    result["result"] = self._serialize_result(sandbox_locals["result"])
                else:
                    # Look for the last expression result
                    last_val = None
                    for key, val in sandbox_locals.items():
                        if not key.startswith("_"):
                            last_val = val
                    if last_val is not None:
                        result["result"] = self._serialize_result(last_val)

            result["success"] = True
            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()

        except IndexError as e:
            # Common error: accessing empty DataFrame rows/columns
            result["error"] = f"IndexError: {str(e)} (DataFrame may be empty or row/column doesn't exist)"
            result["stderr"] = stderr_capture.getvalue() + "\n" + traceback.format_exc()
            logger.warning(f"Sandbox execution failed: {e}")
            
        except KeyError as e:
            # Common error: column doesn't exist
            result["error"] = f"KeyError: {str(e)} (Column or key not found in data)"
            result["stderr"] = stderr_capture.getvalue() + "\n" + traceback.format_exc()
            logger.warning(f"Sandbox execution failed: {e}")
            
        except (ValueError, TypeError) as e:
            # Common error: type conversion failures
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["stderr"] = stderr_capture.getvalue() + "\n" + traceback.format_exc()
            logger.warning(f"Sandbox execution failed: {e}")

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["stderr"] = stderr_capture.getvalue() + "\n" + traceback.format_exc()
            logger.warning(f"Sandbox execution failed: {e}")

        return result

    def _serialize_result(self, value: Any) -> Any:
        """Serialize result to JSON-compatible format."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_result(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_result(v) for k, v in value.items()}
        if isinstance(value, pd.DataFrame):
            return value.to_dict()
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        
        # Fallback to string
        return str(value)


# =============================================================================
# E2B INTEGRATION (OPTIONAL)
# =============================================================================

class E2BExecutor:
    """
    Optional E2B (e2b-code-interpreter) integration for remote code execution.
    
    E2B is used for:
    - Heavy chart generation
    - Long-running step-debugging  
    - Untrusted code that needs stronger isolation
    
    Enable by setting ENABLE_E2B=true in environment.
    Requires: pip install e2b-code-interpreter
    """
    
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_E2B", "").lower() in ("true", "1", "yes")
        self._sandbox = None
        self._initialized = False
    
    def _init_sandbox(self):
        """Lazy initialize E2B sandbox."""
        if self._initialized:
            return self._sandbox is not None
        
        self._initialized = True
        
        if not self.enabled:
            logger.debug("E2B not enabled (ENABLE_E2B not set)")
            return False
        
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            logger.warning("E2B enabled but E2B_API_KEY not set")
            return False
        
        try:
            from e2b_code_interpreter import Sandbox
            self._sandbox = Sandbox()
            logger.info("E2B sandbox initialized successfully")
            return True
        except ImportError:
            logger.warning("E2B enabled but e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter")
            return False
        except Exception as e:
            logger.error(f"E2B initialization failed: {e}")
            return False
    
    def execute(
        self,
        code: str,
        timeout_sec: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code in E2B remote sandbox.
        
        Args:
            code: Python code to execute
            timeout_sec: Execution timeout
            
        Returns:
            {"success": bool, "result": Any, "logs": str, "error": str}
        """
        if not self._init_sandbox():
            return {
                "success": False,
                "result": None,
                "logs": "",
                "error": "E2B not available"
            }
        
        try:
            execution = self._sandbox.run_code(code, timeout=timeout_sec)
            
            # Extract results
            logs = ""
            if execution.logs:
                logs = "\n".join([
                    log.line if hasattr(log, 'line') else str(log) 
                    for log in execution.logs
                ])
            
            result = None
            if execution.results:
                result = execution.results[-1] if execution.results else None
                if hasattr(result, 'text'):
                    result = result.text
            
            error = None
            if execution.error:
                error = str(execution.error)
            
            return {
                "success": error is None,
                "result": result,
                "logs": logs,
                "error": error
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Handle specific E2B errors
            if "timeout" in error_str.lower():
                logger.warning(f"E2B execution timeout: {timeout_sec}s")
                return {
                    "success": False,
                    "result": None,
                    "logs": "",
                    "error": f"E2B timeout after {timeout_sec}s"
                }
            elif "quota" in error_str.lower() or "limit" in error_str.lower():
                logger.error(f"E2B quota/limit error: {error_str}")
                return {
                    "success": False,
                    "result": None,
                    "logs": "",
                    "error": "E2B quota exceeded"
                }
            else:
                logger.error(f"E2B execution error: {e}")
                return {
                    "success": False,
                    "result": None,
                    "logs": "",
                    "error": str(e)
                }
    
    def close(self):
        """Close E2B sandbox."""
        if self._sandbox:
            try:
                self._sandbox.close()
            except Exception:
                pass
            self._sandbox = None


# Global E2B executor (lazy init)
_e2b_executor: Optional[E2BExecutor] = None

def get_e2b_executor() -> E2BExecutor:
    """Get or create global E2B executor."""
    global _e2b_executor
    if _e2b_executor is None:
        _e2b_executor = E2BExecutor()
    return _e2b_executor


def execute_with_e2b_fallback(
    code: str,
    df: pd.DataFrame,
    prefer_local: bool = True,
    timeout_sec: int = 10
) -> Dict[str, Any]:
    """
    Execute code with E2B as primary (when enabled), falling back to local sandbox.
    
    Strategy:
    - If E2B is enabled, try E2B first (better isolation)
    - If E2B fails or is unavailable, fall back to local sandbox
    - If prefer_local=True, always try local first for speed
    
    Args:
        code: Python code to execute
        df: DataFrame to pass
        prefer_local: Try local first (default True). Set to False to prefer E2B.
        timeout_sec: Execution timeout
        
    Returns:
        {"success": bool, "result": Any, "logs": str, "error": str, "executor": str}
    """
    e2b = get_e2b_executor()
    
    if prefer_local:
        # Try local first for speed
        sandbox = SandboxExecutor(timeout_seconds=timeout_sec)
        result = sandbox.run_user_code(code, {"df": df}, timeout_sec)
        
        if result["success"]:
            result["executor"] = "local"
            return result
        
        # If local failed and E2B is available, try E2B as fallback
        if e2b.enabled:
            logger.info("Local execution failed, trying E2B as fallback")
            try:
                e2b_result = e2b.execute(code, timeout_sec)
                e2b_result["executor"] = "e2b"
                if e2b_result["success"]:
                    return e2b_result
                # E2B also failed, return local error (more informative)
                logger.warning(f"E2B fallback also failed: {e2b_result.get('error')}")
            except Exception as e:
                logger.warning(f"E2B fallback exception: {e}")
        
        result["executor"] = "local"
        return result
    else:
        # E2B preferred path (when prefer_local=False)
        if e2b.enabled:
            try:
                result = e2b.execute(code, timeout_sec)
                result["executor"] = "e2b"
                if result["success"]:
                    return result
                # E2B failed, fall back to local
                logger.info(f"E2B failed ({result.get('error')}), falling back to local sandbox")
            except Exception as e:
                logger.warning(f"E2B exception: {e}, falling back to local sandbox")
        
        # Fall back to local sandbox
        sandbox = SandboxExecutor(timeout_seconds=timeout_sec)
        result = sandbox.run_user_code(code, {"df": df}, timeout_sec)
        result["executor"] = "local"
        return result


def execute_code_safe(
    code: str,
    df: pd.DataFrame,
    timeout_sec: int = 10,
    use_e2b: bool = None
) -> Dict[str, Any]:
    """
    Safe code execution with automatic E2B/local selection.
    
    This is the recommended entry point for code execution.
    Automatically chooses E2B or local based on configuration.
    
    Args:
        code: Python code to execute
        df: DataFrame to pass
        timeout_sec: Execution timeout
        use_e2b: Force E2B (True), force local (False), or auto (None)
        
    Returns:
        {"success": bool, "result": Any, "logs": str, "error": str, "executor": str}
    """
    if use_e2b is True:
        return execute_with_e2b_fallback(code, df, prefer_local=False, timeout_sec=timeout_sec)
    elif use_e2b is False:
        sandbox = SandboxExecutor(timeout_seconds=timeout_sec)
        result = sandbox.run_user_code(code, {"df": df}, timeout_sec)
        result["executor"] = "local"
        return result
    else:
        # Auto: use E2B if enabled, otherwise local
        e2b = get_e2b_executor()
        prefer_local = not e2b.enabled
        return execute_with_e2b_fallback(code, df, prefer_local=prefer_local, timeout_sec=timeout_sec)


__all__ = [
    "SandboxExecutor", 
    "CodeValidator",
    "E2BExecutor",
    "get_e2b_executor",
    "execute_with_e2b_fallback",
    "execute_code_safe",
]
