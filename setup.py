from setuptools import setup, find_packages

setup(
    name="ars-dev",
    version="1.0.0",
    description="Adaptive Rejection Sampling for STAT 243 Final Project, Fall 2024",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aparimit Kasliwal, Treves Li, Yuyang Wang",
    url="https://github.berkeley.edu/ap-kasliwal/ars-dev",  # GitHub repo link
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)