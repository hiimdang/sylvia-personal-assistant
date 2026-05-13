from setuptools import setup, find_packages
import os

# Get the long description from the README file
def get_long_description():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        return f.read()

setup(
    name="sylvia-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "langchain-classic",
        "qdrant-client==1.14.1",
        "python-dotenv",
        "sentence-transformers",
        "numpy",
        "flask",
        "torch",
        "fastapi",
        "uvicorn",
        "fastembed",
        "pydantic_ai",
        "pydantic>=2.0.0",
        "pydantic-settings",
        "langchain_community",
        "google-api-python-client",
    ],
    author="hdang",
    author_email="haidanglear@gmail.com",
    description="Core components for the Sylvia RAG agent",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/hiimdang/sylvia-personal-assistant", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
