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
        "pyyaml",
        "requests",
        "loguru",
        "fastapi",
        "uvicorn",
        "aiohttp",
        "setuptools",
        "pydantic",
        "httpx[http2]",
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