from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CrewWatch",
    version="0.0.1",
    author="mayankiitp",
    author_email="msinghmayank62@gmail.com",
    description="A Python library for AI monitoring that tracks resource usage, performance metrics, and operational health in real-time, designed for ease of integration and extensibility.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mayank-iitp/CrewWatch",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "psutil",
        "matplotlib",
        "streamlit",
        "tiktoken"
    ],
    include_package_data=True,
    package_data={
        # Include any additional files here
    },
)