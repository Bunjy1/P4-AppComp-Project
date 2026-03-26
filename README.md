# P4 Applied Computing Lab Project - 2758304H

This is a package for P4 Applied Computing Labs, containing all work pertaining to my unsupervised machine learning (UML) data clean-up project. The project aim was to assess the usefulness in applying UML to clean-up of a 4D azimuthal electron diffraction dataset from a crystal lattice, with the majority of the project work using singular value decomposition (SVD) for clean-up and results appearing promising. Included are files containing the functions designed for the project with comprehensive docstring documentation, and example notebooks showing the mathematical basis and operation of each of the key functions. The package has been designed primarily for use on the UofG Juoyterhub server, but should function on other python platforms (eg. conda), with installation details listed below.

## Contents

Within this repository are all the standard formatting options to allow installation such as this README, a .toml install folder, and licensing information. All code is contained within the /src/ACPP_Bunjy filepath, and appears as follows:

* __init__.py: Reference file for .toml to initiate package functions installation.
* compare.py: Unifying function utilising entire clean-up and fitting process designed for the project, uses functions from both decomp.py and tfp.py
* decomp.py: Functions for data formatting, decomposition-based cleanup of dataset using Hyperspy package, and assessment of clean-up quality with metrics.
* tfp.py: Functions for fitting two-fold periodic functions iteratively to a 4D dataset, and related plotting function.
* weeks123_functions.py: Testbed functions from the first three weeks of lab work.

## Dependencies

This package was designed for use in Python versions 3.9 or higher. Dependencies for the package are as follows (previous versions may operate but the package was created with these packages used, proceed with caution):

* numpy>=2.0.2
* matplotlib>=3.9.2
* scipy>=1.13.1
* scikit-learn>=1.6.1
* tqdm>=4.67.2
* hyperspy>=2.3.0

## Installation (UofG Jupyter)

1. For installation on the university Jupyterhub, the following line in a terminal will fully install the package:

    pip install git+"https://github.com/Bunjy1/P4-AppComp-Project"

2. Optionally test successful installation with (look for package name listed):

   pip list

3. The package should now be installed. Functions from the package can be installed as follows:

    import ACPP_Bunjy.tfp as tfp
    import ACPP_Bunjy.decomp as decomp
    import ACPP_Bunjy.compare as compare

## Installation (Conda)

As the project was desgined for use in the UofG Jupyter, success is not guaranteed with this method. However, this installation should perform as expected.

Prerequisites:
- Conda (Anaconda or Miniconda)

1. If git not already installed in the environment being used, install with:

   conda install git

2. Install the package as above using pip. Conda install will not work for this installation.

    pip install git+"https://github.com/Bunjy1/P4-AppComp-Project"

3. Optionally test successful installation with (look for package name listed):

   pip list

4. The package should now be installed. Functions from the package can be installed as follows:

    import ACPP_Bunjy.tfp as tfp
    import ACPP_Bunjy.decomp as decomp
    import ACPP_Bunjy.compare as compare

