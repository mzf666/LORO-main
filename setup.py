from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="loro-torch",
    version="1.0",
    description="LORO: Parameter and Memory Efficient Pretraining via Low-rank Riemannian Optimization",
    url="https://github.com/mzf666/LORO-main",
    author="Zhanfeng Mo",
    author_email="zhanfeng001@ntu.edu.sg",
    license="Apache 2.0",
    packages=["loro_torch"],
    install_requires=required,
)
