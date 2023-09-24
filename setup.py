from setuptools import setup, find_packages

setup(
    name="unga_speeches",
    version="0.0.1",
    description="",
    url="https://github.com/darenasc/un-speeches/",
    author="Diego Arenas",
    author_email="darenasc@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
