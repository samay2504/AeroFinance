import sys
import os
import logging

logger = logging.getLogger(__name__)

def apply_dll_fix():
    """
    Apply Windows DLL path fix for torch/spacy.
    Must be called before importing torch, spacy, or other dependent libraries.
    """
    if sys.platform == 'win32':
        # Allow duplicate OpenMP libraries (common issue with torch+numpy)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Add torch lib to DLL search path
        # Try multiple common locations for torch lib
        candidate_paths = [
            os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib'),
            os.path.join(os.path.dirname(sys.executable), 'site-packages', 'torch', 'lib'),
        ]
        
        for torch_lib in candidate_paths:
            if os.path.exists(torch_lib):
                # Add to PATH
                os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
                
                # Add to python DLL directory
                try:
                    os.add_dll_directory(torch_lib)
                except (AttributeError, OSError):
                    pass
                
                # FORCE LOAD DLLs using ctypes (Production Fix for WinError 1114)
                try:
                    import ctypes
                    # Order matters: c10 -> torch_cpu -> torch_python
                    dlls_to_load = ['c10.dll', 'torch_cpu.dll', 'torch_python.dll', 'asmjit.dll', 'fbgemm.dll']
                    for dll_name in dlls_to_load:
                        dll_path = os.path.join(torch_lib, dll_name)
                        if os.path.exists(dll_path):
                            try:
                                ctypes.CDLL(dll_path)
                                logger.info(f"Successfully loaded {dll_name} via ctypes")
                            except Exception as dll_err:
                                logger.debug(f"Failed to load {dll_name}: {dll_err}")
                    
                    logger.info(f"Applied DLL fix with ctypes for: {torch_lib}")
                    return True
                except Exception as e:
                    logger.warning(f"DLL fix injection failed: {e}")

        logger.debug("Torch DLL path not found in standard locations")
        return False
