# Design Otimization using Ising Machine
In this project we implement the optimization algorithm in [1] on an optoelectronic Ising machine that we built similar to the one in [2]. The difference is that we implement on a different hardware setup as opposed to D-Wave quantum annealers which were used as the Ising solver in [1]. The optimization problem that we chose is 2D optimization of photonics devices.<br />.

The dataset used here are of metagragratings fro 2D freeform deflector ar 1000nm wavelength and angle range of 45 degrees. These datasets were taken from Metanet[3] and were also generated using ceviche[4].<br />


The Binary Variational Autoencoder (bVAE) used here is structured as: x input layers -> hidden layer(512 nodes)-> hidden layer (xx)-> bottleneck layer(500??)-> gumbell layer (500??)-> hidden layer(xx)-> hidden layer(512)->output layer(x)<br />



References: <br />
1. [Machine Learning Framework for Quantum Sampling of HighlyConstrained, Continuous Optimization Problems ](https://aip.scitation.org/doi/10.1063/5.0060481)<br />
2. [A poor man’s coherent Ising machine based on opto-electronic feedback systems for solving optimization problems](https://www.nature.com/articles/s41467-019-11484-3)<br />
3. [Metanet](http://metanet.stanford.edu/)<br />
4. [ceviche package](https://github.com/fancompute/ceviche)<br /><br />

