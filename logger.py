import logging
from config.default import get_cfg_defaults

logger = logging.getLogger("flowclas")
logger.setLevel(logging.INFO)

class ColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[92m',    # green
        'WARNING': '\033[93m', # yellow
        'ERROR': '\033[91m',   # red
        'RESET': '\033[0m'
    }
    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        return f"{color}{msg}{reset}"

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s: %(message)s", "%m/%d %H:%M:%S"))

file_handler = logging.FileHandler(get_cfg_defaults().SYSTEM.OUTPUT_DIR + "/flowclas.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%m/%d %H:%M:%S"))

logger.handlers = [console_handler, file_handler]