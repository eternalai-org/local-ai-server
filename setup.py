from local_ai import __version__
from setuptools import setup, find_packages

setup(
    name="local_ai",
    version=__version__,
    packages=find_packages(),
    package_data={
        "local_ai": [
            "examples/templates/*.jinja",
            "examples/best_practices/*.json",
        ],
    },
    include_package_data=True,
    install_requires=[
        "httpx[http2]",
        "nvidia-ml-py3",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "loguru>=0.6.0",
        "psutil>=5.9.0",
        "httpx>=0.23.0",
        "lighthouseweb3>=0.1.0",
        "python-dotenv>=0.20.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "aiohttp>=3.8.0",
        "setuptools>=65.0.0",
        "pydantic>=1.9.0",
        "pillow>=9.0.0",
        "asyncio>=3.4.3"
    ],
    entry_points={
        "console_scripts": [
            "local-ai = local_ai.cli:main",
        ],
    },
    author="EternalAI",
    description="A library to manage local language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)