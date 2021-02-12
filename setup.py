from setuptools import setup, find_packages

setup(
    name="blazingma",
    version="0.1.0",
    description="Blazing Multi-Agents: a collection of algorithms applied in multi-agent environments",
    author="Filippos Christianos",
    url="https://github.com/semitable/blazing-ma",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["hydra-core", "torch", "cpprb"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
