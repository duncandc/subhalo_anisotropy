#!/usr/bin/env python

#Duncan Campbell
#February, 2015
#Yale University
#create (sub-)halo catalogues where the subhaloes have been redistributed with haloes

#load packages
from __future__ import print_function, division
import numpy as np
import math
import custom_utilities as cu
import matplotlib.pyplot as plt
import h5py
import sys
from halotools import sim_manager
from halotools import mock_observables
from halotools.utils import aggregation, distance
import transformations

def main():
    
    if len(sys.argv)==1:
        catalogue = 'Bolshoi_250'
    else:
        catalogue = sys.argv[1]
    
    #load halo catalogue
    halocat = sim_manager.CachedHaloCatalog(simname=catalogue, redshift=0.0,
                                            version_name='1.0', halo_finder='Rockstar') 
    halo_table = halocat.halo_table
    Lbox = halocat.Lbox
    N = len(halo_table)
    
    #tabulate the coordinates of the host
    def ag_func(x):
        return x[0]
    sorting_keys = ['halo_hostid', 'halo_upid']
    
    # x-coordinate
    key_list = ['halo_x']
    aggregation.add_new_table_column(halo_table,'host_x','float','halo_hostid',ag_func,key_list,sorting_keys=sorting_keys)
    
    # y-coordinate
    key_list = ['halo_y']
    aggregation.add_new_table_column(halo_table,'host_y','float','halo_hostid',ag_func,key_list,sorting_keys=sorting_keys)
    
    # z-coordinate
    key_list = ['halo_z']
    aggregation.add_new_table_column(halo_table,'host_z','float','halo_hostid',ag_func,key_list,sorting_keys=sorting_keys)
    
    #define host and sub haloes
    host = halo_table['halo_upid'] == -1
    sub = halo_table['halo_upid'] != -1
    
    #calculate the radial distance from host
    host_coords = np.vstack((halo_table['host_x'],halo_table['host_y'],halo_table['host_z'])).T
    sub_coords = np.vstack((halo_table['halo_x'],halo_table['halo_y'],halo_table['halo_z'])).T
    halo_table['local_r'] = distance(host_coords, sub_coords, period=np.array([Lbox]*3))
    
    #calculate the coordinates wrt to the host
    halo_table['local_x'] = halo_table['halo_x']-halo_table['host_x']
    halo_table['local_y'] = halo_table['halo_y']-halo_table['host_y']
    halo_table['local_z'] = halo_table['halo_z']-halo_table['host_z']
    
    flip = np.fabs(halo_table['local_x'])>Lbox/2.0
    halo_table['local_x'][flip] = halo_table['local_x'][flip] - np.sign(halo_table['local_x'][flip])*Lbox
    flip = np.fabs(halo_table['local_y'])>Lbox/2.0
    halo_table['local_y'][flip] = halo_table['local_y'][flip] - np.sign(halo_table['local_y'][flip])*Lbox
    flip = np.fabs(halo_table['local_z'])>Lbox/2.0
    halo_table['local_z'][flip] = halo_table['local_z'][flip] - np.sign(halo_table['local_z'][flip])*Lbox
    
    #calculate the local spherical coordinates
    halo_table['local_theta'] = np.arccos(halo_table['local_z'] / halo_table['local_r'])
    halo_table['local_phi'] = np.arctan2(halo_table['local_y'], halo_table['local_x'])
    
    #remove NaNs, r=0
    halo_table['local_theta'] = np.nan_to_num(halo_table['local_theta'])
    halo_table['local_phi'] = np.nan_to_num(halo_table['local_phi'])
    
    #as a test, calculate the positons using the spherical coordinates
    #calculate new positions
    x = halo_table['host_x'] + halo_table['local_r']*np.sin(halo_table['local_theta'])*np.cos(halo_table['local_phi'])
    y = halo_table['host_y'] + halo_table['local_r']*np.sin(halo_table['local_theta'])*np.sin(halo_table['local_phi'])
    z = halo_table['host_z'] + halo_table['local_r']*np.cos(halo_table['local_theta'])
    
    #take care of PBCs
    flip = (x>Lbox)
    x[flip] = x[flip]-Lbox
    flip = (x<0.0)
    x[flip] = x[flip]+Lbox
    
    flip = (y>Lbox)
    y[flip] = y[flip]-Lbox
    flip = (y<0.0)
    y[flip] = y[flip]+Lbox
    
    flip = (z>Lbox)
    z[flip] = z[flip]-Lbox
    flip = (z<0.0)
    z[flip] = z[flip]+Lbox
    
    #add to table
    halo_table['halo_x_test'] = x
    halo_table['halo_y_test'] = y
    halo_table['halo_z_test'] = z
    halo_table['halo_x_test'][host] = halo_table['halo_x'][host]
    halo_table['halo_y_test'][host] = halo_table['halo_y'][host]
    halo_table['halo_z_test'][host] = halo_table['halo_z'][host]
    
    #randomize orientation of individual subhaloes
    u = np.random.random(N)
    v = np.random.random(N)
    halo_table['random_phi'] = 2.0*np.pi*u
    halo_table['random_theta'] = np.arccos(2.0*v-1.0)
    
    #calculate new positions
    x = halo_table['host_x'] + halo_table['local_r']*np.sin(halo_table['random_theta'])*np.cos(halo_table['random_phi'])
    y = halo_table['host_y'] + halo_table['local_r']*np.sin(halo_table['random_theta'])*np.sin(halo_table['random_phi'])
    z = halo_table['host_z'] + halo_table['local_r']*np.cos(halo_table['random_theta'])
    
    #take care of PBCs
    flip = (x>Lbox)
    x[flip] = x[flip]-Lbox
    flip = (x<0.0)
    x[flip] = x[flip]+Lbox
    
    flip = (y>Lbox)
    y[flip] = y[flip]-Lbox
    flip = (y<0.0)
    y[flip] = y[flip]+Lbox
    
    flip = (z>Lbox)
    z[flip] = z[flip]-Lbox
    flip = (z<0.0)
    z[flip] = z[flip]+Lbox
    
    #add to table
    halo_table['halo_x_ran1'] = x
    halo_table['halo_y_ran1'] = y
    halo_table['halo_z_ran1'] = z
    halo_table['halo_x_ran1'][host] = halo_table['halo_x'][host]
    halo_table['halo_y_ran1'][host] = halo_table['halo_y'][host]
    halo_table['halo_z_ran1'][host] = halo_table['halo_z'][host]
    
    #randomly rotate system
    unq_ids, N_mem = np.unique(halo_table['halo_hostid'], return_counts=True)
    N_sys = len(unq_ids)
    """
    u = np.random.random(N_sys)
    v = np.random.random(N_sys)
    phi_rot = 2.0*np.pi*u
    theta_rot = np.arccos(2.0*v-1.0)
    
    phi_rot = np.repeat(phi_rot, N_mem)
    theta_rot = np.repeat(theta_rot, N_mem)
    
    new_phis = halo_table['local_phi']+phi_rot
    new_thetas = halo_table['local_theta']+theta_rot
    """
    
    print('here')
    #create rotation matrices
    Rs = [transformations.random_rotation_matrix()[:3,:3] for x in range(0,N_sys)]
    Rs = np.repeat(Rs, N_mem, axis=0)
    local_positions = np.vstack((halo_table['local_x'],halo_table['local_y'],halo_table['local_z'])).T
    N = len(local_positions)
    new_local_positions = np.array([np.dot(local_positions[i],Rs[i].T) for i in range(0,N)])
    print('here,here')
    
    print(local_positions)
    print(new_local_positions)
    
    new_local_x = new_local_positions[:,0]
    new_local_y = new_local_positions[:,1]
    new_local_z = new_local_positions[:,2]
    
    x = halo_table['host_x'] + new_local_x
    y = halo_table['host_y'] + new_local_y
    z = halo_table['host_z'] + new_local_z
    
    
    """
    #calculate new positions
    x = halo_table['host_x'] + halo_table['local_r']*np.sin(new_thetas)*np.cos(new_phis)
    y = halo_table['host_y'] + halo_table['local_r']*np.sin(new_thetas)*np.sin(new_phis)
    z = halo_table['host_z'] + halo_table['local_r']*np.cos(new_thetas)
    """
    
    #take care of PBCs
    flip = (x>Lbox)
    x[flip] = x[flip]-Lbox
    flip = (x<0.0)
    x[flip] = x[flip]+Lbox
    
    flip = (y>Lbox)
    y[flip] = y[flip]-Lbox
    flip = (y<0.0)
    y[flip] = y[flip]+Lbox
    
    flip = (z>Lbox)
    z[flip] = z[flip]-Lbox
    flip = (z<0.0)
    z[flip] = z[flip]+Lbox
    
    #add to table
    halo_table['halo_x_ran2'] = x
    halo_table['halo_y_ran2'] = y
    halo_table['halo_z_ran2'] = z
    halo_table['halo_x_ran2'][host] = halo_table['halo_x'][host]
    halo_table['halo_y_ran2'][host] = halo_table['halo_y'][host]
    halo_table['halo_z_ran2'][host] = halo_table['halo_z'][host]
    
    
    #save table
    #set some properties
    redshift = 0.0
    Lbox = 250.0
    particle_mass = 1.35*10**8.0
    num_halos = len(halo_table)
    simname = 'Bolshoi_250'
    halo_finder='Rockstar'
    version_name = 'isotropized_1.0'
    processing_notes = 'Catalog only contains (sub-)halos with mpeak mass greater than 100 particles.'
    
    #create custom catalogue
    halo_table = np.array(halo_table)
    halo_catalog = sim_manager.UserSuppliedHaloCatalog(redshift = redshift,
                                           Lbox = Lbox,
                                           particle_mass = particle_mass,
                                           halo_x = halo_table['halo_x'],
                                           halo_y = halo_table['halo_y'],
                                           halo_z = halo_table['halo_z'],
                                           halo_vx = halo_table['halo_vx'],
                                           halo_vy = halo_table['halo_vy'],
                                           halo_vz = halo_table['halo_vz'],
                                           halo_id = halo_table['halo_id'],
                                           halo_pid = halo_table['halo_pid'],
                                           halo_upid = halo_table['halo_upid'],
                                           halo_mvir = halo_table['halo_mvir'],
                                           halo_m200b = halo_table['halo_m200b'],
                                           halo_m200c = halo_table['halo_m200c'],
                                           halo_rvir = halo_table['halo_rvir'],
                                           halo_rs = halo_table['halo_rs'],
                                           halo_vmax = halo_table['halo_vmax'],
                                           halo_first_acc_scale = halo_table['halo_first_acc_scale'],
                                           halo_acc_scale = halo_table['halo_acc_scale'],
                                           halo_mpeak = halo_table['halo_mpeak'],
                                           halo_vpeak = halo_table['halo_vpeak'],
                                           halo_mpeak_scale = halo_table['halo_mpeak_scale'],
                                           halo_vmax_at_mpeak = halo_table['halo_vmax_at_mpeak'],
                                           halo_half_mass_scale = halo_table['halo_half_mass_scale'],
                                           halo_x_test = halo_table['halo_x_test'],
                                           halo_y_test = halo_table['halo_y_test'],
                                           halo_z_test= halo_table['halo_z_test'],
                                           halo_x_ran1 = halo_table['halo_x_ran1'],
                                           halo_y_ran1 = halo_table['halo_y_ran1'],
                                           halo_z_ran1 = halo_table['halo_z_ran1'],
                                           halo_x_ran2 = halo_table['halo_x_ran2'],
                                           halo_y_ran2 = halo_table['halo_y_ran2'],
                                           halo_z_ran2 = halo_table['halo_z_ran2'],
                                           halo_r = halo_table['local_r'],
                                           halo_phi = halo_table['local_phi'],
                                           halo_theta = halo_table['local_theta'],
                                           halo_local_x = halo_table['local_x'],
                                           halo_local_y = halo_table['local_y'],
                                           halo_local_z = halo_table['local_z']
                                           )
    
    #save the catalogue
    fname = cu.get_output_path() + 'processed_data/Multidark/Bolshoi/halo_catalogues/Bolshoi_250_isotropized.hdf5'
    halo_catalog.add_halocat_to_cache(fname, simname, halo_finder,
                                      version_name, processing_notes, overwrite=True)


def rotation_matrix(angle, direction):
    """
    Return matrix to rotate about axis defined by point and direction.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    
    return M

if __name__ == '__main__':
    main()