function [ AbsEff ] = Abs_Eff( Pattern, wavelength, target_angle,pol)



input_angle=0;
Wavelength =wavelength; 
Polarization=pol;

Target=[1];% Target diffraction order in the x-direction for each polarization i.e. [1 -1]
nBot = 1.45;% refractive index of the substrate
nTop = 1;% refractive index of the top...air
nDevice =3.48;
Thickness=325;

Grid=90;
ZGrid=50;

SymmetryY = 1; % Enforces symmetry in the Y direction
SymmetryX = 0; % Enforces symmetry in the X direction

Period = [Wavelength*Target/(sind(target_angle)-sind(input_angle)),0.5*Wavelength];
Fourier=[12 12];



%Compute incident k-vector
kParallelForward = sind(input_angle);

% Compute total Fourier orders
NFourier = ceil(Fourier.*Period/Wavelength);

% Define polarization values
if strcmp(Polarization,'TE') 
    Polarizations = 1;
elseif strcmp(Polarization,'TM')
    Polarizations = -1;
elseif strcmp(Polarization,'Both')
    Polarizations = [1, -1];
else
    error('Invalid polarization');
end
NumPol = length(Polarizations); 

% Define grid for the device
[xGrid, yGrid, GridScale] = DefineGrid_dataset(target_angle, Wavelength); %xgrid and ygrid are the center cordinates of each box
Nx = length(xGrid); %Number of x grid points
Ny = length(yGrid); %Number of y grid points


% Define full device stack
DeviceProfile = {[0,Thickness,0],[1,3,2]}; % See Reticolo documentaion for definitions

%Initialize Reticolo
retio([],inf*1i);


% Define textures for each layer
LayerTextures = cell(1,3);
LayerTextures{1} = {nTop};
LayerTextures{2} = {nBot};
nPattern = Pattern*(nDevice - nTop) + nTop;
LayerTextures{3} = FractureGeom(nPattern,nTop,nDevice,xGrid,yGrid);

% Initialize empty field matrix
FieldProductWeighted = zeros(NumPol,ZGrid,Nx,Ny);

% Begin polarization loop
% Can be changed to parfor as necessary
for polIter = 1:NumPol  
    % Set simulation parameters in Reticolo

    
    ReticoloParm = res0;
    if SymmetryX || SymmetryY
        ReticoloParm.sym.pol = Polarizations(polIter);
        if SymmetryX
            ReticoloParm.sym.x = Period(1)/2;
        end
        if SymmetryY
            ReticoloParm.sym.y = Period(2)/2;
        end
    end
    ReticoloParm.res3.npts = [0,ZGrid,0]; %Number of points to sample field in each layer
    ReticoloParm.res3.sens = -1; % Default to illumination from below
    ReticoloParm.res1.champ = 1; % Accurate fields
    
    

    % res1 computes the scattering matrices of each layer
    LayerResults = res1(Wavelength,Period,LayerTextures,NFourier,kParallelForward,0,ReticoloParm);

    % res2 computes the scattering matrix of the full device stack
    DeviceResults = res2(LayerResults,DeviceProfile);


    if (Polarizations(polIter)==1) %For TE polarization
        % Extract simulation results
        TransmittedLight = DeviceResults.TEinc_bottom_transmitted;
        
        % Find appropriate target
        TargetIndex = find((TransmittedLight.order(:,1)==Target(polIter))&(TransmittedLight.order(:,2)==0));
        
        % Store efficiencies
        AbsEff = TransmittedLight.efficiency_TE(TargetIndex);

        
    elseif (Polarizations(polIter)==-1) %For TM polarization
        % Extract simulation results
        TransmittedLight = DeviceResults.TMinc_bottom_transmitted;
        
        % Find appropriate target and store efficiencies
        TargetIndex = find((TransmittedLight.order(:,1)==Target(polIter))&(TransmittedLight.order(:,2)==0));
        AbsEff = TransmittedLight.efficiency_TM(TargetIndex);

    end

end
