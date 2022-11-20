% Compute the simulation grid for given geometry
function [xGrid, yGrid, dr] = DefineGrid_dataset(target_angle, Wavelength)
    
    %Number of grid points
    
    Nx = 256;
    Ny = 128;
    
    %Device period
    Px = Wavelength/sind(target_angle);
    Py = Wavelength/2;
    
    %Compute external grid coordinates
    xBounds = linspace(0,Px,Nx+1); 
    yBounds = linspace(0,Py,Ny+1);
    
    %Compute size of each grid box
    dx = xBounds(2) - xBounds(1);
    dy = yBounds(2) - yBounds(1);
    
    %Compute coordinates of center of each box
    xGrid = xBounds(2:end)- 0.5*dx;
    yGrid = yBounds(2:end)- 0.5*dy;
    
    %Compute average grid size
    dr = mean([dx dy]);
end
