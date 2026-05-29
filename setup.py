from setuptools import find_packages, setup

with open("requirements.txt") as fd:
    install_requires = fd.read().splitlines()

setup(
    name="tox21full",
    version="0.1.0",
    description="Generate the full Tox21 multi-task toxicity dataset from PubChem BioAssay records",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    keywords=[
        "tox21",
        "pubchem",
        "bioassay",
        "toxicity",
        "cheminformatics",
        "molecular-machine-learning",
        "dataset",
    ],
    author="JJ Ben-Joseph",
    author_email="jj@memoriesofzion.com",
    python_requires=">=3.8",
    url="https://github.com/agentcures/tox21full",
    project_urls={
        "Source": "https://github.com/agentcures/tox21full",
        "Issues": "https://github.com/agentcures/tox21full/issues",
    },
    license="Apache-2.0",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={"console_scripts": ["tox21full = tox21full.__main__:main"]},
)
