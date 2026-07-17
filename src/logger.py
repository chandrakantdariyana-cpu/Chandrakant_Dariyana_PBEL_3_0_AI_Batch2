import logging
from pathlib import Path

log_path = Path("logs/log.log")

log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s || %(levelname)s || %(filename)s || %(lineno)s || %(message)s"
)

logger = logging.getLogger("CyberGuardAI")
logger.setLevel(logging.INFO)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)