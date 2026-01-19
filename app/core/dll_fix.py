import sys
import os
import logging

logger = logging.getLogger(__name__)

_dll_fix_applied = False

def apply_dll_fix():
    """
    Apply Windows DLL path fix for torch/spacy.
    Must be called BEFORE importing torch, spacy, transformers, langchain, or other dependent libraries.
    
    This fix addresses WinError 1114 on Windows with conda environments.
    """
    global _dll_fix_applied
    
    if _dll_fix_applied:
        return True
    
    if sys.platform != 'win32':
        _dll_fix_applied = True
        return True
    
    # Allow duplicate OpenMP libraries (common issue with torch+numpy)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Prevent transformers from loading torch on import
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    # Build comprehensive list of candidate paths
    candidate_paths = []
    
    # 1. Conda environment (highest priority)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidate_paths.append(os.path.join(conda_prefix, 'Lib', 'site-packages', 'torch', 'lib'))
    
    # 2. Current interpreter's site-packages
    for path in sys.path:
        if 'site-packages' in path:
            torch_lib = os.path.join(path, 'torch', 'lib')
            if torch_lib not in candidate_paths:
                candidate_paths.append(torch_lib)
    
    # 3. Standard Python locations
    candidate_paths.extend([
        os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib'),
        os.path.join(os.path.dirname(sys.executable), 'site-packages', 'torch', 'lib'),
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'),
    ])
    
    for torch_lib in candidate_paths:
        if os.path.exists(torch_lib):
            # Add to PATH environment variable
            current_path = os.environ.get('PATH', '')
            if torch_lib not in current_path:
                os.environ['PATH'] = torch_lib + os.pathsep + current_path
            
            # Add to DLL search directories (Python 3.8+)
            try:
                os.add_dll_directory(torch_lib)
            except (AttributeError, OSError):
                pass
            
            # FORCE LOAD DLLs using ctypes (Production Fix for WinError 1114)
            try:
                import ctypes
                # Order matters: dependencies first
                dlls_to_load = [
                    'c10.dll', 
                    'asmjit.dll', 
                    'fbgemm.dll',
                    'torch_cpu.dll', 
                    'torch_python.dll'
                ]
                loaded_count = 0
                for dll_name in dlls_to_load:
                    dll_path = os.path.join(torch_lib, dll_name)
                    if os.path.exists(dll_path):
                        try:
                            ctypes.CDLL(dll_path)
                            loaded_count += 1
                            logger.debug(f"Loaded {dll_name}")
                        except Exception as dll_err:
                            logger.debug(f"Could not load {dll_name}: {dll_err}")
                
                if loaded_count > 0:
                    logger.info(f"DLL fix applied: {torch_lib} ({loaded_count} DLLs loaded)")
                    _dll_fix_applied = True
                    return True
                    
            except Exception as e:
                logger.warning(f"DLL fix failed for {torch_lib}: {e}")
    
    logger.debug("Torch DLL path not found - torch may not be installed")
    _dll_fix_applied = True  # Mark as applied to avoid repeated attempts
    return False


def is_dll_fix_applied() -> bool:
    """Check if DLL fix has been applied."""
    return _dll_fix_applied

