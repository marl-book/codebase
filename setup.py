from setuptools import setup, find_packages

setup(
    name="marlbase",
    version="0.1.0",
    description="Fast Multi-Agent RL: a collection of algorithms applied in multi-agent environments",
    author="Filippos Christianos, Lukas Schäfer",
    url="https://github.com/marl-book/codebase",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["hydra-core>=1.1", "torch", "cpprb", "einops"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
