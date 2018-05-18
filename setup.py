from setuptools import setup, find_packages
setup(
    name="Fast-Parapred",
    version="1.0",
    packages= ["aid25"],
    entry_points={
        "console_scripts": ['fast_parapred = aid25.library_commands:main']
    },
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "torch>=0.3",
        "pandas>=0.19.2,<0.20",
        "numpy>=1.14.2",
        "matplotlib>=2.0.0",
        "scikit-learn>=0.18,<0.19",
        "scipy>=0.19",
        "biopython==1.69",
        "docopt>=0.6.2",
    ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.csv'],
        "aid25": ["data/*.csv", "cv-ab-seq/ag_weights.pth.tar","cv-ab-seq/rnn_weights.pth.tar",
                  "cv-ab-seq/parapred_weights.pth.tar", "cv-ab-seq/atrous_self_weights.pth.tar" ]
    },

    # metadata for upload to PyPI
    author="Andreea-Ioana Deac",
    author_email="aid25@cam.ac.uk",
    description="Neural architectures for end-to-end paratope prediction",
    keywords="paratope prediction antigen antibody binding",
    url="https://github.com/andreeadeac22/part2",   # project code
)
