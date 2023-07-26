import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import nlopt
import argparse


def main(args):
    seed = args.seed #np.random.randint(100000)
    np.random.seed(seed)
    mp.verbosity(0)


    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)
    Air = mp.Medium(index=1.0)

    
    design_region_length = args.design_region_length # 15 # (μm)
    design_region_width = args.design_region_width # 4 # (μm)
    design_region_thickness= args.design_region_thickness # 0.22

    waveguide_width = args.waveguide_width #0.5 # (μm)
    waveguide_length = args.waveguide_length #0.5 # (μm)
    waveguide_thickness = design_region_thickness

    soi_thickness = 1
    soi_length = 2*waveguide_length + design_region_length
    soi_width = design_region_width

    air_cladding=200 # just some large value

    resolution = args.resolution #20 # (pixels/μm)

    arm_separation = design_region_width-waveguide_width #3.5 # (μm) distance between arms center to center
    pml_size = 1.0 # (μm)

    minimum_length = args.minimun_length # 0.09 # (μm)
    eta_e = 0.75
    #filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length,eta_e) # (μm)

    filter_radius=mpa.get_conic_radius_from_eta_e(minimum_length,eta_e)

    eta_i = 0.5
    eta_d = 1-eta_e
    design_region_resolution = int(resolution) # (pixels/μm)
    frequencies = 1/np.linspace(1.5,1.6,5) # (1/μm)

    Nx = int(design_region_resolution*design_region_length)
    Ny = int(design_region_resolution*design_region_width)
    Nz = int(design_region_resolution*design_region_thickness)

    design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),Air,Si,grid_type="U_MEAN", do_averaging = False)### not giving a Nz because we want it to extent in the z axis so you give the z direction width in the volumn
    design_region = mpa.DesignRegion(design_variables,
        volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_length, design_region_width, design_region_thickness)))###
        
    Sx = 2*pml_size + 2*waveguide_length + design_region_length+1 # cell size in X
    Sy = 2*pml_size + design_region_width + 1 # cell size in Y
    Sz = 2*pml_size + design_region_thickness + soi_thickness+ 1 # cell size in Z ###
        
    cell_size = mp.Vector3(Sx,Sy,Sz)

    pml_layers = [mp.PML(pml_size)]

    fcen = 1/args.wavelength
    width = 0.2
    fwidth = width * fcen
    #source_center  = [-Sx/2 + pml_size + waveguide_length/3,arm_separation/2,0]
    source_size    = mp.Vector3(0,2*waveguide_width,2*waveguide_thickness)
    kpoint = mp.Vector3(1,0,0)

    source_center1  = [-Sx/2 + pml_size + waveguide_length/3,arm_separation/2,0]
    source_center2  = [-Sx/2 + pml_size + waveguide_length/3,-arm_separation/2,0]

    src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
    sources = [mp.EigenModeSource(src,
                        eig_band = 1,
                        direction=mp.NO_DIRECTION,
                        eig_kpoint=kpoint,
                        size = source_size,
                        center=source_center1,
                        amplitude=1.0),
                mp.EigenModeSource(
                        src,
                        eig_band=1,
                        direction=mp.NO_DIRECTION,
                        eig_kpoint=kpoint,
                        size=source_size,
                        center=source_center2,
                        amplitude=0.0 - 1.0j
        )]



    # src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
    # source = [mp.EigenModeSource(src,
    #                     eig_band = 1,
    #                     direction=mp.NO_DIRECTION,
    #                     eig_kpoint=kpoint,
    #                     size = source_size,
    #                     center=source_center)]


    geometry=[]

    # the top and bottom cladding
    geometry.append(mp.Block(center=mp.Vector3(0, 0, -(waveguide_thickness+soi_thickness)/2), material=SiO2, size=mp.Vector3(mp.inf, mp.inf, soi_thickness))) #Oxide layer
    geometry.append(mp.Block(center=mp.Vector3(0, 0, (-waveguide_thickness+air_cladding)/2), material=Air, size=mp.Vector3(mp.inf, mp.inf, air_cladding))) #the air cladding

    # the waveguides
    geometry.append(mp.Block(center=mp.Vector3(x=-(design_region_length+waveguide_length)/2, y=arm_separation/2), material=Si, size=mp.Vector3(Sx/2+1, waveguide_width, waveguide_thickness))) # top left waveguide
    geometry.append(mp.Block(center=mp.Vector3(x=-(design_region_length+waveguide_length)/2, y=-arm_separation/2), material=Si, size=mp.Vector3(Sx/2+1, waveguide_width, waveguide_thickness))) # bottom left waveguide
    geometry.append(mp.Block(center=mp.Vector3(x=(design_region_length+waveguide_length)/2, y=arm_separation/2), material=Si, size=mp.Vector3(Sx/2+1, waveguide_width, waveguide_thickness))) # top right waveguide
    geometry.append(mp.Block(center=mp.Vector3(x=(design_region_length+waveguide_length)/2, y=-arm_separation/2), material=Si, size=mp.Vector3(Sx/2+1, waveguide_width, waveguide_thickness))) # bottom right waveguide

    # the design region
    geometry.append(mp.Block(center=design_region.center, size=design_region.size, material=design_variables))


    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        #symmetries=[mp.Mirror(direction=mp.Y)],
                        default_material=SiO2,
                        resolution=resolution,
                        eps_averaging = True,)


    def mapping(x,eta,beta):
        
        x = (npa.fliplr(x.reshape(Nx,Ny)) + x.reshape(Nx,Ny))/2 # up-down symmetry
        x = (npa.flipud(x.reshape(Nx,Ny)) + x.reshape(Nx,Ny))/2 # left-right symmetry
        
        # filter
        filtered_field = mpa.conic_filter(x, filter_radius,design_region_length,design_region_width,design_region_resolution)
        
        #filtered_field = mpa.cylindrical_filter(x, filter_radius,design_region_length,design_region_width,design_region_resolution)
        
        # projection
        projected_field = mpa.tanh_projection(filtered_field,beta,eta)
        
        # interpolate to actual materials
        return projected_field.flatten()



    mode = 1
    motinor_z=0.5

    TE0 = mpa.EigenmodeCoefficient(sim,
            mp.Volume(center=mp.Vector3(x=Sx/2 - pml_size - 2*waveguide_length/3, y=arm_separation/2),
                size=mp.Vector3(y=2* waveguide_width, z=motinor_z)),mode)
    TE_top = mpa.EigenmodeCoefficient(sim,
            mp.Volume(center=mp.Vector3(-Sx/2 + pml_size + 2*waveguide_length/3,arm_separation/2,0),
                size=mp.Vector3(y=2* waveguide_width, z=motinor_z)),mode)
    TE_bottom = mpa.EigenmodeCoefficient(sim,
            mp.Volume(center=mp.Vector3(-Sx/2 + pml_size + 2*waveguide_length/3,-arm_separation/2,0),
                size=mp.Vector3(y=2* waveguide_width, z=motinor_z)),mode)
    ob_list = [TE0,TE_top,TE_bottom]

    def J(output,top_source,bottom_source):
        power = npa.abs(output/top_source) ** 2 + npa.abs(output/bottom_source) ** 2 
        return npa.mean(power)

    opt = mpa.OptimizationProblem(
        simulation = sim,
        objective_functions = J,
        objective_arguments = ob_list,
        design_regions = [design_region],
        frequencies=frequencies
    )




    evaluation_history = []
    cur_iter = [0]
    def f(v, gradient, cur_beta):
        print("Current iteration: {}".format(cur_iter[0]+1))
        
        f0, dJ_du = opt([mapping(v,eta_i,cur_beta)])
    
        if gradient.size > 0:
            gradient[:] = tensor_jacobian_product(mapping,0)(v,eta_i,cur_beta,np.sum(dJ_du,axis=1))
        
        evaluation_history.append(np.max(np.real(f0)))

        cur_iter[0] = cur_iter[0] + 1
        
        return np.real(f0)


    algorithm = nlopt.LD_MMA
    n = Nx * Ny # number of parameters

    # Initial guess
    x = np.ones((n,)) * 0.5 # + np.random.normal(loc=mean, scale=std, size=(n,))
    #x = np.random.rand(n,) 

    #just for the same of boundary value
    x=np.where(x>1,1,x)
    x=np.where(x<0,0,x)

    # lower and upper bounds
    lb = 0
    ub = 1

    cur_beta = args.cur_beta #4
    beta_scale = args.beta_scale #2
    num_betas = args.num_betas #6
    update_factor = args.update_factor #30
    for iters in range(num_betas):
        print("current beta: ",cur_beta)
        
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a,g: f(a,g,cur_beta))
        solver.set_maxeval(update_factor)
        x[:] = solver.optimize(x)
        cur_beta = cur_beta*beta_scale



    update_factor = 5
    num_beta=6
    print("----------------------------------------------------------------")
    print('final binarization')
        
    for iters in range(num_betas): # the main beta increment loop
        print("current beta: ",cur_beta)
        
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a,g: f(a,g,cur_beta))
        solver.set_maxeval(update_factor)
        x[:] = solver.optimize(x) # iterates update_factor times
        cur_beta = cur_beta*beta_scale

    sim_data_string=str(design_region_length)+"x"+str(design_region_width)+"_"+str(cur_beta)+"_"+str(beta_scale)+"_"+str(num_betas)+"_"+str(update_factor)

    evaluation_history = np.array(evaluation_history)
    np.save("evaluation_history_"+sim_data_string+".npy",x)

    np.save("3d_dc"+sim_data_string+".npy",x)
    # extract evaluation_history and plot 10*np.log10(0.5*np.array(evaluation_history))








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=float, default=4929, help='seed for the numpy random generator')
    parser.add_argument('-waveguide_length', type=float, default=0.5, help='length of the waveguide in micrometeter')
    parser.add_argument('-waveguide_width', type=float, default=0.5, help='width of the waveguide in micrometeter')
    parser.add_argument('-design_region_length', type=float, default=7.0, help='length of the design in micrometeter')
    parser.add_argument('-design_region_width', type=float, default=3.0, help='width of the design in micrometeter')
    parser.add_argument('-minimum_length', type=float, default=0.09, help='minimum feature length in micrometer')
    parser.add_argument('-wavelength', type=float, default=1.55, help='the wavelength for the simulation')
    parser.add_argument('-cur_beta', type=float, default=4, help='starting value of beta')
    parser.add_argument('-beta_scale', type=float, default=2, help='the factor by which beta is incremented (multiplied)')
    parser.add_argument('-num_betas', type=int, default=6, help='number of beta increments')
    parser.add_argument('-update_factor', type=int, default=30, help='optimization iterations per beta value')

    args = parser.parse_args()
    main(args)


'''
seed=1000 ..
waveguide_width=0.5 ..
design_region_length=7 ..
design_region_width=3 ..
design_region_thickness= 0.22
waveguide_length=0.5 ..
minimum_length=0.09 ..
wavelength=1.55 ..
cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 30


'''