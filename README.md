# P4 Applied Computing Lab Project - 2758304H

This is a package for P4 Applied Computing Labs, containing all work pertaining to my unsupervised machine learning (UML) data clean-up project. The project aim was to assess the usefulness in applying UML to clean-up of an azimuthal electron diffraction dataset from a crystal lattice, with the majority of the project work using singular value decomposition (SVD) for clean-up and results appearing promising. Included are files containing the functions designed for the project with comprehensive docstring documentation, and example notebooks showing the mathematical basis and operation of each of the key functions. The package has been designed primarily for use on the UofG Juoyterhub server, but should function on other python platforms (eg. conda), with installation details listed below.

## Contents

Within this repository are all the standard formatting options to allow installation such as this README, a .toml install folder, and licensing information. All code is contained within the /src/ACPP_Bunjy filepath, and appears as follows:

* Week

## Dependencies

## Installation (UofG Jupyter)

## Installation (Conda)

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
