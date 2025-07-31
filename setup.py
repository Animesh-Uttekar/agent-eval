from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agenteval",
    version="0.1.0",
    description="A modular Python SDK for evaluating AI agent prompts and outputs with various metrics and LLM-as-a-Judge.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Animesh Uttekar",
    author_email="animeshuttekar98@gmail.com",
    url="https://github.com/Animesh-Uttekar/agent-eval",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "langchain",
        "transformers",
        "scikit-learn",
        "numpy",
        "nltk",
        "tqdm",
        "python-dotenv",
        "rouge-score",
        "evaluate",
        "httpx",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
