import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure simple logging to both console and file."""
    handler_console = logging.StreamHandler(sys.stdout)

    fmt = "[%(levelname)s] %(asctime)s %(name)s:%(lineno)d — %(message)s"
    formatter = logging.Formatter(fmt)

    handler_console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler_console)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

