import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eventx",
    version="0.0.1",
    author="Marc Hübner",
    author_email="marc.hbnr@gmail.com",
    description="Jointly extracts multiple events from natural language text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marchbnr/eventx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)
