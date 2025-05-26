import sys
import argparse
from loguru import logger
from local_ai import __version__
from local_ai.config import CONFIG
from local_ai.utils import health_check


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for managing local large language models"
    )
    parser.add_argument(
        "--version", action="version", version=f"Local AI version: {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest='command', help="Commands for managing local language models"  
    )
    status_command = subparsers.add_parser(
       "status", help="Check the running model"
    )
    return parser.parse_known_args()

def version_command():
    logger.info(
        f"Local AI version: {__version__}"
    )

def handle_status(args):
    proxy_port = CONFIG["proxy_port"]

    if health_check(f"http://localhost:{proxy_port}"):
        logger.info(f"Proxy is healthy")
    else:
        logger.error(f"Proxy is not healthy")
        return None
    
    model_hash = CONFIG["model"]["hash"]
    print(model_hash)
    return model_hash
    

def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        logger.error(f'unknown command or argument: {arg}')
        sys.exit(2)

    if known_args.command == "version":
        version_command()
    elif known_args.command == "status":
        handle_status(known_args)
    else:
        logger.error(f"Unknown command: {known_args.command}")
        sys.exit(2)


if __name__ == "__main__":
    main()