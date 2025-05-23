import logging
from yaml import load, Loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""Local AI - A library to manage local AI models."""
__version__ = "2.0.0"

CONFIG = load(open("configs/8x4090.yaml", "r"), Loader=Loader)