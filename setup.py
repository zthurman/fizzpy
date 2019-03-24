import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fizzpy",
    version="0.01",
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
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD+Patent",
        "Operating System :: OS Independent",
    ],
)
