"""
Sandbox Executor - Secure Python code execution with AST validation.
Runs LLM-generated code in restricted environment.
"""
import logging
import ast
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
    """

    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout_seconds
        self.max_memory = max_memory_mb
        self.validator = CodeValidator()

    def _clean_code(self, code: str) -> str:
        """Clean code of markdown artifacts and formatting issues."""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python", 1)[-1]
        if "```" in code:
            code = code.split("```")[0]

        # Remove leading/trailing whitespace
        code = code.strip()

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

        return code

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


__all__ = ["SandboxExecutor", "CodeValidator"]
