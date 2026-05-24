from setuptools import setup, find_packages

setup(
    name="agentops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "qdrant-client>=1.6.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "agentops=agentops.cli:main",
        ],
    },
)
