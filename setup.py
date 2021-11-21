from setuptools import setup, find_packages

setup(
    name="fastmarl",
    version="0.1.0",
    description="Fast Multi-Agent RL: a collection of algorithms applied in multi-agent environments",
    author="Filippos Christianos",
    url="https://github.com/semitable/fast-marl",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["hydra-core>=1.1", "torch", "cpprb", "einops"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
