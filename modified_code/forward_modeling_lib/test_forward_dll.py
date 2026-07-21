from pathlib import Path
import ctypes


dll_path = Path(__file__).resolve().parent / "libpyforward.dll"

if not dll_path.exists():
    raise FileNotFoundError(dll_path)

lib = ctypes.CDLL(str(dll_path))

lib.test_function.argtypes = [ctypes.c_int]
lib.test_function.restype = ctypes.c_int

result = lib.test_function(7)

print("DLL path:", dll_path)
print("test_function(7):", result)