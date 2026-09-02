from setuptools import setup, find_packages

setup(
    name="mcfile-python",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[],
    author="mx-cnie",
    author_email="developer@metex-tech.com",
    description="A basic Python wrapper for the MACA mcFile API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MetaX-MACA/mcfile-python",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
