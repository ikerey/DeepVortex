# This script displays in a figure the output of DeepVortex and the horizotal 
# velocity field overlaid with streamlines.  

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Frame interval
ninit=0  #initial frame
nfin=16  #final frame

# Files and paths
file_vel = 'sample.fits'
file_vor = 'output.fits'
file_output = 'vortex_detection_DeepVortex'
path_vel = "/DeepVortex/Sample/"
path_vor = "/DeepVortex/Output/"
path_output = "/DeepVortex/Figs/"

## ----------------------- Constants  Plot -------------------------- #
# Pixel size in Mm
scale = 39.875e-3  # escala Mm/pixel 
# Distance to allocate cbar
cbdist = 0.01  
# Thickness to allocate cbar
cbthick = 0.02  
# Lower and upper limit
rangePlot = [-1,1]


## ------------------------- Read Data ----------------------------- ##
def read_data(path):
    """Read input parameters"""
    f = fits.open(path)
    ima = f[0].data
    return ima


vel = read_data(path_vel+file_vel)
vor = read_data(path_vor+file_vor)

vx = vel[0,:,:,0]
vy = vel[0,:,:,1]

szx=vy[0,:].size
szy =vx[:,0].size
szt=nfin-ninit+1

Y,X = np.ogrid[:szy,:szx]
x = np.arange(0, szx, 1)
y = np.arange(0, szy, 1)
Xc,Yc = np.meshgrid(x,y)

## ---------------------------- Plot ------------------------------- ##

for i in np.arange(ninit,nfin+1,1):
    dcn = '%03d'%(i)
    print('frame: ', i)

    vx = vel[i,:,:,0]
    vy = vel[i,:,:,1]

    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot()
    plt.imshow(vor[i,:,:,0], cmap='RdBu', extent=[0,szx*scale,0,szy*scale], vmin=rangePlot[0],vmax=rangePlot[1])
    plt.autoscale(False)
    plt.streamplot(X*scale, Y*scale, vx, vy, density=10, linewidth=0.75, arrowsize=0.75, arrowstyle='->', color='k')   
    plt.contour(Xc*scale, Yc*scale, np.absolute(vor[i,:,:,0]),[0.5], colors='lime')    
    plt.xlabel('[Mm]')
    plt.ylabel('[Mm]')
    plt.minorticks_on()
    #Plot color bar
    plt.imshow(vor[i,:,:,0], cmap='RdBu', extent=[0,szx*scale,0,szy*scale], vmin=rangePlot[0],vmax=rangePlot[1])       
    box = ax.get_position()
    cbaxes = plt.axes([box.x0 + box.width + cbdist, box.y0, cbthick, box.height])
    cb = plt.colorbar(cax=cbaxes, orientation="vertical")
    cb.ax.minorticks_on()


    # Save and show plot
    plt.savefig(path_output+file_output+'_'+dcn+'.png',bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.close()
## ----------------------------------------------------------------- ##
