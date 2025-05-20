import sys
import asyncio
import argparse
from pathlib import Path
from loguru import logger
from local_ai import __version__
from local_ai.core import LocalAIManager
from local_ai.upload import upload_folder_to_lighthouse
from local_ai.utils import check_downloading
from local_ai.download import check_downloaded_model, download_model_from_filecoin_async

manager = LocalAIManager()

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
    start_command = subparsers.add_parser(
        "start", help="Start a local language model server"
    )
    start_command.add_argument(
        "--hash", type=str, required=True,
        help="Filecoin hash of the model to start"
    )
    start_command.add_argument(
        "--port", type=int, default=8080,
        help="Port number for the local language model server"
    )
    start_command.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host address for the local language model server"
    )
    start_command.add_argument(
        "--context-length", type=int, default=32768,
        help="Context length for the local language model server"
    )
    stop_command = subparsers.add_parser(
        "stop", help="Stop a local language model server"
    )
    download_command = subparsers.add_parser(
       "download", help="Download and extract model files from IPFS"
    )
    download_command.add_argument(
        "--hash", required=True,
        help="IPFS hash of the model metadata"
    )
    download_command.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Chunk size for downloading files"
    )
    download_command.add_argument(
        "--output-dir", type=Path, default = None,
        help="Output directory for model files"
    )
    upload_command = subparsers.add_parser(
        "upload", help="Upload model files to IPFS"
    )
    upload_command.add_argument(
        "--folder-name", type=str, required=True,
        help="Folder containing model files"
    )
    upload_command.add_argument(
        "--model-family", type=str, required=False,
        help = "Model family (e.g., GPT-3, GPT-4, etc.)"
    )   
    upload_command.add_argument(
        "--zip-chunk-size", type=int, default=512,
        help="Chunk size for splitting compressed files"
    )
    upload_command.add_argument(
        "--threads", type=int, default=16,
        help="Number of threads for compressing files"
    )
    upload_command.add_argument(
        "--max-retries", type=int, default=20,
        help="Maximum number of retries for uploading files"
    )
    upload_command.add_argument(
        "--hf-repo", type=str, default = None,
        help="Hugging Face model repository"
    )
    upload_command.add_argument(
        "--hf-file", type=str, default = None,
        help="Hugging Face model file"
    )
    upload_command.add_argument(
        "--ram", type=float, default=None,
        help="RAM in GB for the serving model at 4096 context length"
    )
    check_command = subparsers.add_parser(
        "check", help="Model metadata check"
    )
    check_command.add_argument(
        "--hash", type=str, required=True,
        help="Model name to check existence"
    )
    status_command = subparsers.add_parser(
       "status", help="Check the running model"
    )
    restart_command = subparsers.add_parser(
        "restart", help="Restart the local language model server"
    )
    check_downloading_command = subparsers.add_parser(
        "downloading", help="Check if the model is being downloaded"
    )
    return parser.parse_known_args()

def version_command():
    logger.info(
        f"Local AI version: {__version__}"
    )

def handle_download(args):
    asyncio.run(download_model_from_filecoin_async(args.hash))

def handle_start(args):
    if not manager.start(args.hash, args.port, args.host, args.context_length):
        sys.exit(1)

def handle_stop(args):
    if not manager.stop():
        sys.exit(1)
    
def handle_check(args):
    is_downloaded = check_downloaded_model(args.hash)
    res = "True" if is_downloaded else "False"
    print(res)
    return res

def handle_status(args):
    running_model = manager.get_running_model()
    if running_model:
        print(running_model)

def handle_upload(args):
    kwargs = {
        "family": args.model_family,
        "ram": args.ram,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
    }
    upload_folder_to_lighthouse(args.folder_name, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)

def handle_restart(args):
    if not manager.restart():
        sys.exit(1)

def handle_check_downloading(args):
    downloading_files = check_downloading()
    if not downloading_files:
        return False
    str_files = ",".join(downloading_files)
    print(str_files)
    return True

def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        logger.error(f'unknown command or argument: {arg}')
        sys.exit(2)

    if known_args.command == "version":
        version_command()
    elif known_args.command == "start":
        handle_start(known_args)
    elif known_args.command == "stop":
        handle_stop(known_args)
    elif known_args.command == "download":
        handle_download(known_args)
    elif known_args.command == "check":
        handle_check(known_args)
    elif known_args.command == "status":
        handle_status(known_args)
    elif known_args.command == "upload":
        handle_upload(known_args)
    elif known_args.command == "restart":
        handle_restart(known_args)
    elif known_args.command == "downloading":
        handle_check_downloading(known_args)
    else:
        logger.error(f"Unknown command: {known_args.command}")
        sys.exit(2)


if __name__ == "__main__":
    main()