## Overview

PLEASE NOTE THAT THE `evaluation_history` plot is wrong in these notebooks...made a silly mistake<br/><br/>

In the notebooks here, I have tried to implement adjoint optimization for designing a Silicon Photonic devices.<br/><br/>

For designing devices which are phase sensitive such as the directional coupler, a "Phase injected Topology Optimization" technique [1] was used. In the 2D optimization you can see that for different initial conditions (permittivity distributions) we get different designs, though not necessarily with good results. For the 3D optimization a computing cluster is needed ( the 3D splitter took over night on a MacBook for just 4-5 iterations ). The 3D optimization of the 3dB directional  coupler needs to be done in a computing cluster. Meep in mind the FDTD is not compute intentive but it is need very high memory-bandwidth.<br/>

*NOTE*: In the Optimization loop, for every beta_num, onlt the initial condition is cur_beta for optimization. Its not that the cur_beta is applied again and again because in that case it bust binarizes with in few epochs where as in the main optimization loop it doesn't. Verify with the following code:<br/>

```
initial_des=np.random.rand(Nx,Ny)

for i in range(20):
    initial_des=mapping(initial_des,0.5,2)
    design_region.update_design_parameters(initial_des)
sim.plot2D()
```

Also, plotting of insertion loss(dB)


The "Sandbox" notebooks are just few examples which I created when I was learning to use MEEP for adjoint optimization.

*TODO*: Try to use [ceviche](https://github.com/fancompute/ceviche) and [Angler](https://github.com/fancompute/angler/tree/master) to generate the device designs butget the figure-of-merit from [MEEP](https://github.com/NanoComp/meep)

## References
1. [Phase-Injected Topology Optimization for Scalable and Interferometrically Robust Photonic Integrated Circuits - Alec M. Hammond, Joel B. Slaby, Michael J. Probst, and Stephen E. Ralph ACS Photonics 2023](https://pubs.acs.org/doi/10.1021/acsphotonics.2c01016)


