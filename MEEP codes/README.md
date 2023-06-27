## Overview

In the notebooks here, I have tried to implement adjoint optimization for designing a Silicon Photonic devices.<br/><br/>

For designing devices which are phase sensitive such as the directional coupler, a "Phase injected Topology Optimization" technique [1] was used. In the 2D optimization you can see that for diierent initial conditions (permittivity distributions) we get different designs, though not necessarily with good results. For the 3D optimization a computing cluster is needed ( the 3D splitter took over night on a MacBook for just 4-5 iterations ). The 3D optimization of the 3dB directional  coupler needs to be done in a computing cluster the speed ut up.<br/><br/>

The "Sandbox" notebooks are just few examples which I tried when I was learning to use MEEP for adjoint optimization.

*TODO*: Try to use [ceviche](https://github.com/fancompute/ceviche) and [Angler](https://github.com/fancompute/angler/tree/master) to generate the device designs butget the figure-of-merit from [MEEP](https://github.com/NanoComp/meep)

## References
1. [Phase-Injected Topology Optimization for Scalable and Interferometrically Robust Photonic Integrated Circuits - Alec M. Hammond, Joel B. Slaby, Michael J. Probst, and Stephen E. Ralph ACS Photonics 2023](https://pubs.acs.org/doi/10.1021/acsphotonics.2c01016)


