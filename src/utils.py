from typing import IO, Any
import os

def safe_open(file_path: str, mode: str) -> IO[Any]:
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	return open(file_path, mode)
