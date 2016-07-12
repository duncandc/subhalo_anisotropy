#!/usr/bin/env python

#Duncan Campbell
#February, 2015
#Yale University
#examine the alignments of subhaloes

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
    
    keep = halo_table['halo_mpeak']>10**12
    halo_table = halo_table[keep]
    
    print(halo_table.dtype.names)
    
    sub = halo_table['halo_upid']!=-1
    
    
    
    
    

if __name__ == '__main__':
    main()
