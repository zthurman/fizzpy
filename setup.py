import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fizzpy",
    version="0.0.2",
    author="Zechariah Thurman",
    author_email="zechariah.thurman@gmail.com",
    description="A math library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zthurman/fizzpy",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)