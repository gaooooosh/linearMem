from setuptools import setup, find_packages

setup(
    name="qwen3mem",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # 指定 src 目录
)
