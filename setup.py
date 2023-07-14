import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustutils4r", # Replace with your own username
    version="0.0.6",
    author="Rutuja Gurav",
    author_email="rutujagurav100@gmail.com",
    description="Wrapper around some basic sklearn utilities for clustering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rutujagurav/clustutils4r",
    project_urls={
        "Bug Tracker": "https://github.com/rutujagurav/clustutils4r/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'seaborn', 'opentsne', 'pyclustertend', 'scikit-learn'
      ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)