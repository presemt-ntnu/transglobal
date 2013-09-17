# Setup virtual python environment based on Anaconda python distribution
#
# Make sure the Anaconda binary path, containing the "conda" command, is in your PATH
#
# To remove the virtual environment use something like: 
# $rm -r ~/anaconda/envs/transglobal
#
# The easiest way to update the virtual environment is simply to delete the
# old one and to create a new one (under the same name).

# Create virtual environment with required packages
ENV_NAME="transglobal"
CONDA_PACKAGES="h5py ipython ipython-notebook matplotlib networkx nose numpy pip pydot scikit-learn"
conda create -n "$ENV_NAME" $CONDA_PACKAGES

# Activate (to use right pip binary)
source activate "$ENV_NAME"

# Install additional packages through pip
PIP_PACKAGES="asciitable configobj suds" 
pip install $PIP_PACKAGES




