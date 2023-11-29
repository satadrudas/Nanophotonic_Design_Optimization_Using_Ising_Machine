# Inverse Design Otimization using Ising Machine
## Overview
In this project we implement the optimization algorithm in ref [1] on an optoelectronic Ising machine that we built similar to the one in ref [2]. The difference is that we implement on a different hardware setup as opposed to D-Wave quantum annealers which were used as the Ising solver in [1]. Using the opto-electronic Ising Machine is advantageous since all-to-all connections can be implemented with it whereas in D-Wave computer only sparce connections are available. The optimization problem that we chose is 2D optimization of photonics devices.<br /><br/>

The following image illustrates the algorithm in [1]. The changes in our project is we use a CIM instead of the D-Wave anealer. We are optimizing a size constrained Silicon Photonic 3dB directional coulpler.<br/>
![image](https://github.com/satadrudas98/Nanophotonic_Design_Optimization_Using_Ising_Machine/assets/38806771/6b5eb63e-4818-494e-88a3-87d099a43367)


## Dataset generation

The inverse designed dataset is generated from several python based FDTD/FDFD packaged such as [MEEP](https://github.com/NanoComp/meep), [ceviche](https://github.com/fancompute/ceviche), [Angler](https://github.com/fancompute/angler/tree/master) etc. One devide of interest is a silicon photonic 3dB directional coupler, but also datset of defractive metastructures can be optained from [Metanet](http://metanet.stanford.edu/) as well. MEEP was used for final design verification but ceviche and Anglere were used for fast generation of datasets.<br /><br />

In case of generating dataset using MEEP, a computing cluster such as AWS can be used.(to do)<br /><br />

## The Binary Variational Autoencoder architecture (bVAE)

The bVAE used here is structured as: x input layers -> hidden layer(512 nodes)-> hidden layer (xx)-> bottleneck layer(500??)-> gumbell layer (500??)-> hidden layer(xx)-> hidden layer(512)->output layer(x)<br />




## References: <br />
1. [Machine Learning Framework for Quantum Sampling of HighlyConstrained, Continuous Optimization Problems ](https://aip.scitation.org/doi/10.1063/5.0060481)<br />
2. [A poor man’s coherent Ising machine based on opto-electronic feedback systems for solving optimization problems](https://www.nature.com/articles/s41467-019-11484-3)<br />
3. [AWS instance selector](https://d1.awsstatic.com/AWS%20EC2%20for%20HPC%20Solution%20brief%20Final.pdf)


