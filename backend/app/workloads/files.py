import os
import time
from pathlib import Path


def atomic_replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.025 * (attempt + 1))
