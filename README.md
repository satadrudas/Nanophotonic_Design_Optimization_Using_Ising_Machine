# Inverse Design Otimization using Ising Machine
## Overview
In this project we implement the optimization algorithm in ref [1] on an optoelectronic Ising machine that we built similar to the one in ref [2]. The difference is that we implement on a different hardware setup as opposed to D-Wave quantum annealers which were used as the Ising solver in [1]. Using the opto-electronic Ising Machine is advantageous since all-to-all connections can be implemented in it whereas in D-Wave computer only sparce connections are available. The optimization problem that we chose is 2D optimization of photonics devices.<br /><br/>

## Dataset generation

The inverse designed dataset is generated from several python based FDTD/FDFD packaged such as [MEEP](https://github.com/NanoComp/meep), [ceviche](https://github.com/fancompute/ceviche), [Angler](https://github.com/fancompute/angler/tree/master) etc. One devide of interest is a silicon photonic 3dB directional coupler, but also datset of defractive metastructures can be optained from [Metanet](http://metanet.stanford.edu/) as well. MEEP was used for final design verification but ceviche and Anglere were used for fast generation of datasets.<br /><br />

In case of generating dataset using MEEP, a computing cluster such as AWS can be used.(to do)<br /><br />

## The Binary Variational Autoencoder architecture (bVAE)

The bVAE used here is structured as: x input layers -> hidden layer(512 nodes)-> hidden layer (xx)-> bottleneck layer(500??)-> gumbell layer (500??)-> hidden layer(xx)-> hidden layer(512)->output layer(x)<br />



## References: <br />
1. [Machine Learning Framework for Quantum Sampling of HighlyConstrained, Continuous Optimization Problems ](https://aip.scitation.org/doi/10.1063/5.0060481)<br />
2. [A poor man’s coherent Ising machine based on opto-electronic feedback systems for solving optimization problems](https://www.nature.com/articles/s41467-019-11484-3)<br />


