#!/usr/bin/env python

#Duncan Campbell
#February, 2015
#Yale University
#caclulate the clustering of an input catalogue

#load packages
from __future__ import print_function, division
import numpy as np
import custom_utilities as cu
import matplotlib.pyplot as plt
import h5py
import sys
from halotools import sim_manager
from halotools import mock_observables

def main():
    
    if len(sys.argv)==1:
        catalogue = 'Bolshoi_250'
    else:
        catalogue = sys.argv[1]
    
    #load halo catalogue
    halocat = sim_manager.CachedHaloCatalog(simname=catalogue, redshift=0.0, version_name='isotropized_1.0', halo_finder='Rockstar') 
    halo_table = halocat.halo_table
    Lbox = halocat.Lbox
    
    #define radial bins
    rbins = np.logspace(-1,1.5,20)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0
    
    selection = (halo_table['halo_mpeak']>10**12.0)
    print("number of gals in selection: {0}".format(np.sum(selection)))
    
    coords = np.vstack((halo_table['halo_x'],halo_table['halo_y'],halo_table['halo_z'])).T
    xi_1 = mock_observables.tpcf(coords[selection], rbins=rbins, period=Lbox)
    xi_1_1h, xi_1_2h = mock_observables.tpcf_one_two_halo_decomp(coords[selection],
        halo_table['halo_hostid'][selection], rbins=rbins, period=Lbox)
    
    coords = np.vstack((halo_table['halo_x_ran1'],halo_table['halo_y_ran1'],halo_table['halo_z_ran1'])).T
    xi_2 = mock_observables.tpcf(coords[selection], rbins=rbins, period=Lbox)
    xi_2_1h, xi_2_2h = mock_observables.tpcf_one_two_halo_decomp(coords[selection],
        halo_table['halo_hostid'][selection], rbins=rbins, period=Lbox)
    
    coords = np.vstack((halo_table['halo_x_ran2'],halo_table['halo_y_ran2'],halo_table['halo_z_ran2'])).T
    xi_3 = mock_observables.tpcf(coords[selection], rbins=rbins, period=Lbox)
    xi_3_1h, xi_3_2h = mock_observables.tpcf_one_two_halo_decomp(coords[selection],
        halo_table['halo_hostid'][selection], rbins=rbins, period=Lbox)
    
    fig1, axes = plt.subplots(figsize=(3.3, 3.3))
    fig1.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot(rbin_centers,xi_1, '-')
    #plt.plot(rbin_centers,xi_1_1h, '-')
    #plt.plot(rbin_centers,xi_1_2h, '-')
    plt.plot(rbin_centers,xi_2, '--')
    plt.plot(rbin_centers,xi_3, ':')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$ \xi $')
    plt.xlabel(r'$[ h^{-1}{\rm Mpc}]$')
    plt.show()
    
    fig2, axes = plt.subplots(figsize=(3.3, 3.3))
    fig2.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot(rbin_centers,rbin_centers*0.0,'-')
    #plt.plot(rbin_centers,(xi_1_1h)/xi_1-1.0, '-')
    #plt.plot(rbin_centers,(xi_1_2h)/xi_1-1.0, '-')
    p1, = plt.plot(rbin_centers,xi_2/xi_1-1.0, '--')
    p2, = plt.plot(rbin_centers,xi_3/xi_1-1.0, ':')
    #plt.plot(rbin_centers,xi_3_1h/xi_1-1.0, ':')
    #plt.plot(rbin_centers,xi_3_2h/xi_1-1.0, ':')
    plt.ylim([-0.1,0.1])
    plt.xscale('log')
    plt.ylabel(r'$\Delta \xi $')
    plt.xlabel(r'$[ h^{-1}{\rm Mpc}]$')
    plt.legend((p1,p2),('individual','system'),loc=4,fontsize=10, frameon=False, labelspacing=0.01)
    plt.show()
    
    fig2, axes = plt.subplots(figsize=(3.3, 3.3))
    fig2.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    #plt.plot(rbin_centers,rbin_centers*0.0,'-')
    plt.plot(rbin_centers,(xi_2_1h)/xi_1_1h-1.0, '--')
    plt.plot(rbin_centers,(xi_2_2h)/xi_1_2h-1.0, '--')
    plt.plot(rbin_centers,(xi_3_1h)/xi_1_1h-1.0, ':')
    plt.plot(rbin_centers,(xi_3_2h)/xi_1_2h-1.0, ':')
    plt.ylim([-0.1,0.1])
    plt.xscale('log')
    plt.ylabel(r'$\Delta \xi $')
    plt.xlabel(r'$[ h^{-1}{\rm Mpc}]$')
    plt.show()
    
    """
    selection = (halo_table['halo_mpeak']>10**12.0) & (halo_table['halo_z']<25)
    plt.figure(figsize=(3.3,3.3))
    plt.plot(halo_table['halo_x'][selection],halo_table['halo_y'][selection],'.', ms=2)
    plt.show()
    
    selection = (halo_table['halo_mpeak']>10**12.0) & (halo_table['halo_z']<25)
    plt.figure(figsize=(3.3,3.3))
    plt.plot(halo_table['halo_x_ran1'][selection],halo_table['halo_y_ran1'][selection],'.', ms=2)
    plt.show()
    """
    
    
if __name__ == '__main__':
    main()