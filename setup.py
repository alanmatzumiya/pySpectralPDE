from pathlib import Path
from setuptools import setup, find_packages
from pySpectralPDE import version

__version__ = version.__version__

BASE_PATH = Path(__file__).resolve().parent

# read the version from the particular file
with open(BASE_PATH / "pySpectralPDE" / "version.py", "r") as f:
    exec(f.read())

DOWNLOAD_URL = f"https://github.com/alanmatzumiya/pySpectralPDE/archive/v{__version__}.tar.gz"

# read the version from the particular file
with open(BASE_PATH / "README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pySpectralPDE",
    package_data={"pySpectralPDE": ["py.typed"]},
    packages=find_packages(),
    zip_safe=False,  # this is required for mypy to find the py.typed file
    version=__version__,
    license="MIT",
    description="Python package for solving PDEs' using spectral methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alan Matzumiya",
    author_email="alan.matzumiya@gmail.com",
    url="https://github.com/alanmatzumiya/pySpectralPDE",
    download_url=DOWNLOAD_URL,
    keywords=["burgers", "partial-differential-equations", "spectral-methods"],
    python_requires=">=3.7",
    install_requires=["matplotlib", "numpy", "numba", "scipy"],
    classifiers=[
        "Development Status :: null",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
