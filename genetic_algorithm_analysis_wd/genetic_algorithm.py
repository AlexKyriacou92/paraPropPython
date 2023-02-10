import datetime
import os
import random
import h5py
import numpy as np
from scipy.interpolate import interp1d
import subprocess
import time
import configparser

import sys

'''
sys.path.append('../genetic_algorithm_analysis/')
from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection
'''

class GA: #Genetic Algorithm Class
    def __init__(self, nGenerations, nIndividuals, nGenes, fRoulette, fTournament, fMutation, fImmigration, fElite):
        # GA Parameters
        self.nGenerations = nGenerations # Number of Generations
        self.nIndividuals = nIndividuals # Number of Individuals per Generation
        self.nGenes = nGenes # Number of Genes per Individual

        # Selection Routines
        self.fRoulette = fRoulette # Fraction of Selection by Roulette
        self.fTournament = fTournament # Fraction of Selection by Tournament

        # Genetic Operators
        self.fCrossOver = 1 - fMutation - fImmigration - fElite
        self.fMutation = fMutation # Fraction of Individuals to undergo Mutation
        self.fImmigrant = fImmigration # Fraction of Individuals to be introduced to a generation via Immigration
        self.fElite = fElite # Fraction of Individuals to be Designated 'Elite'
        self.first_generation = np.zeros((self.nIndividuals, self.nGenes))

    def initialize_from_sample(self, individual_pool):
        sample_of_individuals = random.sample(individual_pool, self.nIndividuals)
        for i in range(self.nIndividuals):
            self.first_generation[i] = sample_of_individuals[i]

def read_from_config(fname_config): # Creates GA using a config file
    config = configparser.ConfigParser()
    config.read(fname_config)
    GA_params = config['GA']
    print('nIndivudals', int(GA_params['nIndividuals']))

    GA_from_config = GA(nGenerations=int(GA_params['nGenerations']), nIndividuals=int(GA_params['nIndividuals']), nGenes = int(GA_params['nGenes']),
                        fRoulette=float(GA_params['fRoulette']), fTournament=float(GA_params['fTournament']),
                        fMutation=float(GA_params['fMutation']), fImmigration=float(GA_params['fImmigration']), fElite=float(GA_params['fElite']))
    return GA_from_config