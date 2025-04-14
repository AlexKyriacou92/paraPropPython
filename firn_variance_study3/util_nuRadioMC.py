import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser

import NuRadioMC
from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

def get_ray_times(x_tx, z_tx, x_rx, z_rx, mode='radiopropa'):
    if mode != 'radiopropa' and mode != 'analytic':
        print('error! mode must be radiopropa or analytic')
        return -1
    else:
        prop = propagation.get_propagation_module(mode)
        ref_index_model = 'greenland_simple'
        ice = medium.get_ice_model(ref_index_model)
        attenuation_model = 'GL1'

        initial_point = [x_tx, 0, -z_tx]
        final_point = [x_rx, 0, -z_rx]
        rays = prop(ice, attenuation_model,
                    n_frequencies_integration=25,
                    n_reflections=0)
        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()

        #sampling_rate_detector = 1 * units.GHz
        #nyquist_frequency = 0.5 * sampling_rate_detector

        travel_times = []
        path_lengths = []
        amp_list = []
        n_surface = 1.3
        n_ice = 1.78
        n_air = 1.0
        solutions = []
        for i_solution in range(rays.get_number_of_solutions()):
            # Or the path length
            path_length = rays.get_path_length(i_solution)
            # And the travel time
            travel_time = rays.get_travel_time(i_solution)
            # print('travel_time', travel_time)
            travel_times.append(travel_time)
            path_lengths.append(path_length)
            solution_int = rays.get_solution_type(i_solution)
            solution_type = solution_types[solution_int]
            solutions.append(solution_type)
            amp_of_path = 1. * (1 / path_length)  # Include Spreading Loss (missing a factor of pi)
            if i_solution == 0:
                amp_of_path *= abs(n_surface - n_air) / abs(n_surface + n_air)  # Apply Fresnel Reflection Coefficient
            amp_list.append(amp_of_path)

        return travel_times, solutions

def get_ray_points(x_tx, z_tx, x_rx, z_rx, mode='radiopropa'):
    if mode != 'radiopropa' and mode != 'analytic':
        print('error! mode must be radiopropa or analytic')
        return -1
    else:
        prop = propagation.get_propagation_module(mode)
        ref_index_model = 'greenland_simple'
        ice = medium.get_ice_model(ref_index_model)
        attenuation_model = 'GL1'

        initial_point = [x_tx, 0, -z_tx]
        final_point = [x_rx, 0, -z_rx]
        rays = prop(ice, attenuation_model,
                    n_frequencies_integration=25,
                    n_reflections=0)
        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()

        #sampling_rate_detector = 1 * units.GHz
        #nyquist_frequency = 0.5 * sampling_rate_detector

        travel_times = []
        path_lengths = []
        amp_list = []
        n_surface = 1.3
        n_ice = 1.78
        n_air = 1.0
        solutions = []
        for i_solution in range(rays.get_number_of_solutions()):
            # Or the path length
            path_length = rays.get_path_length(i_solution)
            # And the travel time
            travel_time = rays.get_travel_time(i_solution)
            # print('travel_time', travel_time)
            travel_times.append(travel_time)
            path_lengths.append(path_length)
            solution_int = rays.get_solution_type(i_solution)
            solution_type = solution_types[solution_int]
            solutions.append(solution_type)
            amp_of_path = 1. * (1 / path_length)  # Include Spreading Loss (missing a factor of pi)
            if i_solution == 0:
                amp_of_path *= abs(n_surface - n_air) / abs(n_surface + n_air)  # Apply Fresnel Reflection Coefficient
            amp_list.append(amp_of_path)

        return travel_times, amp_list, path_lengths, solutions

def ray_paths(x_tx, z_tx, x_rx, z_rx):
    prop = propagation.get_propagation_module('analytic')
    ref_index_model = 'greenland_simple'
    ice = medium.get_ice_model(ref_index_model)
    attenuation_model = 'GL1'

    initial_point = [x_tx, 0, -z_tx]
    final_point = [x_rx, 0, -z_rx]
    rays = prop(ice, attenuation_model,
                n_frequencies_integration=25,
                n_reflections=0)
    rays.set_start_and_end_point(initial_point, final_point)
    rays.find_solutions()

    x_paths = []
    z_paths = []
    solutions = []
    for i_solution in range(rays.get_number_of_solutions()):
        rays_2D = ray_tracing_2D(ice, attenuation_model)
        initial_point_2D = np.array([initial_point[0], initial_point[2]])
        final_point_2D = np.array([final_point[0], final_point[2]])
        C_0 = rays.get_results()[i_solution]['C0']
        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]
        xx, zz = rays_2D.get_path(initial_point_2D, final_point_2D, C_0)
        x_paths.append(xx)
        z_paths.append(zz)
        solution_type = solution_types[solution_int]
        solutions.append(solution_type)
        #solutions.append(solution_type)
    return x_paths, z_paths, solutions

def get_start_theta(x_tx, z_tx, x_rx, z_rx):
    x_paths, z_paths, solutions = ray_paths(x_tx, z_tx, x_rx, z_rx)

    theta_list = []
    for i in range(len(solutions)):
        x_paths_i = x_paths[i]
        z_paths_i = z_paths[i]

        z1 = z_paths_i[1]
        z0 = z_paths_i[0]
        dz = z1 - z0
        x1 = x_paths_i[1]
        x0 = x_paths_i[0]
        dx = x1 - x0
        theta_i = np.arctan2(dz, dx)
        theta_list.append(theta_i)
    return theta_list, solutions