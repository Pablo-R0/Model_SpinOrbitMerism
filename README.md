# SpinOrbitMerism

This repository contains two jupyter notebooks (.ipynb) that explain how to reproduce the data and the figures in the article: "Competition between Spinmerism and Spin-orbit
for a d2 Metal Ion in An Open-Shell Ligand Field" of P. Roseiro, A. Shah, S. Yalouz and V. Robert.

It needs the package QuantNBody (https://github.com/SYalouz/QuantNBody.git). The HOW_TO_INSTALL_QUANTNBODY.txt file guides the installation.

The .py files in /bulk_codes are bulkier versions (for Spyder, e.g.) of the tutorials.




# To install QuantNBody and jupyter-notebook
```
conda create -n quant
conda activate quant

conda install -c psi4 psi4
conda install pip
conda install numpy
pip install matplotlib

git clone -b Replacing_LIL_MATRIX https://github.com/SYalouz/QuantNBody.git
cd QuantNBody
python -m pip install -e .

conda install notebook
```
