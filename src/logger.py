import logging
import os
from datetime import datetime
from pathlib import Path


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
project_root = Path(__file__).resolve().parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = logs_dir / LOG_FILE

logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)

