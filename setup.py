from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('-e .'):
                requirements.append(line)
    return requirements

setup(
    name="BikeDemandForecast",
    version="0.1.0",
    author="BrianMure",
    author_email="b.mureithi40@gmail.com",
    description="A project to forecast bike rental demand using machine learning, including K-Means clustering.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BrianMure/BikeDemandForecast",
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
