from setuptools import setup, find_packages
setup(
    name="fast_parapred",
    version="1.0",
    packages=find_packages(),
    scripts=['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
    },

    # metadata for upload to PyPI
    author="Andreea-Ioana Deac",
    author_email="aid25@cam.ac.uk",
    description="Neural architectures for end-to-end paratope prediction",
    keywords="paratope prediction antigen antibody binding",
    url="https://github.com/andreeadeac22/part2",   # project code
)
