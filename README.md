# P4 Applied Computing Lab Project - 2758304H

This is a package for P4 Applied Computing Labs. You can use
[GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

## Installation

This project is designed to be installed into a conda environment using pip.

Prerequisites:
- Conda (Anaconda or Miniconda)
- Git

1. Clone the repository

    git clone https://github.com/Bunjy1/P4-AppComp-Project.git
    cd P4-AppComp-Project

2. Create and activate a conda environment

    conda create -n appcomp python=3.10
    conda activate appcomp

3. Install required dependencies

    conda install pip git

4. Install the package

Install the project in editable mode (recommended for development):

    pip install -e .

5. (Optional) Use with Jupyter Notebook

If using Jupyter, register the environment as a kernel:

    python -m ipykernel install --user --name appcomp --display-name "Python (appcomp)"

Launch Jupyter and select "Python (appcomp)" as the kernel.

6. Verify installation

In Python or a Jupyter notebook:

    import ACPP_Bunjy

If no error is raised, the installation was successful.
