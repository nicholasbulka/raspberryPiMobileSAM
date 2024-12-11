from setuptools import setup, find_packages

setup(
    name="sam_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "numpy",
        "opencv-python",
        "torch",
        "psutil",
    ],
    python_requires=">=3.8",
)
