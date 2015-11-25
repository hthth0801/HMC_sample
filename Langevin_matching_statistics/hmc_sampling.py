# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:55:38 2015
HMC sampling. This version is different from the tutorial version in theano webpage. 
In this version, we can actually include the hmc with only one step.
@author: user
"""

import theano
import theano.tensor as T
#import numpy


def kinetic_energy(vel):
    """Returns the kinetic energy associated with the given velocity
    and mass of 1.

    Parameters
    ----------
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the kinetic entry associated with vel[i].

    """
    return 0.5 * (vel ** 2).sum(axis=1)
    
def hamiltonian(pos, vel, energy_fn):
    """
    Returns the Hamiltonian (sum of potential and kinetic energy) for the given
    velocity and position.

    Parameters
    ----------
    pos: theano matrix
        Symbolic matrix whose rows are position vectors.
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used tox
        compute the potential energy at a given position.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the Hamiltonian at position pos[i] and
        velocity vel[i].
    """
    # assuming mass is 1
    return energy_fn(pos) + kinetic_energy(vel)
    
def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Performs a Metropolis-Hastings accept-reject move.

    Parameters
    ----------
    energy_prev: theano vector
        Symbolic theano tensor which contains the energy associated with the
        configuration at time-step t.
    energy_next: theano vector
        Symbolic theano tensor which contains the energy associated with the
        proposed configuration at time-step t+1.

    Returns
    -------
    return: accept (theano vector)
        Symbolic theano vector to represent the accepance rate of each sample
    """
    ediff = energy_next - energy_prev
    return T.nnet.sigmoid(-ediff), -ediff
    #return T.minimum(1., T.exp(-ediff)), -ediff
def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    
    def leapfrog(pos, vel, step):
        dE_dpos1 = T.grad(energy_fn(pos).sum(), pos)
        #step_n_1 means we have n*1 vector, in this way, theano will automatically broadcast it to elementwisely multiply with the matrix
        step_n_1 = step.dimshuffle(0,'x')
        new_vel_half = vel - 0.5* step_n_1 * dE_dpos1
        new_pos_full = pos + step_n_1 * new_vel_half
        dE_dpos2 = T.grad(energy_fn(new_pos_full).sum(), new_pos_full)
        new_vel_full = new_vel_half - 0.5* step_n_1 *dE_dpos2
        # from vel(t+stepsize/2) compute pos(t+stepsize)
        
        return [new_pos_full, new_vel_full], {}
        
    (all_pos, all_vel), scan_updates = theano.scan(
        leapfrog,
        outputs_info=[
            dict(initial=initial_pos),
            dict(initial=initial_vel),
        ],
        non_sequences=[stepsize],
        n_steps=n_steps)
    assert not scan_updates
    final_pos = all_pos[1:]
    final_vel = all_vel[1:]
    final_pos1 = all_pos[-1]
    final_vel1 = all_vel[-1]
    return final_pos, final_vel, final_pos1, final_vel1
    """
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    stepsize_n_1 = stepsize.dimshuffle(0,'x')
    vel_half_step = initial_vel - 0.5 * stepsize_n_1 * dE_dpos
    pos_full_step = initial_pos + stepsize_n_1 * vel_half_step
    energy_single = energy_fn(pos_full_step)   
    final_vel = vel_half_step - 0.5 * stepsize_n_1 * T.grad(energy_single.sum(), pos_full_step)
    return pos_full_step, final_vel
    """
    
def hmc_move(initial_vel,positions, energy_fn, stepsize, n_steps):
    final_pos, final_vel, final_pos1, final_vel1 = simulate_dynamics(
        initial_pos=positions,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    accept1, ndeltaH1 = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos1, final_vel1, energy_fn),
    )
    final_pos_vec = T.reshape(final_pos, (final_pos.shape[0]*final_pos.shape[1],final_pos.shape[2]))
    final_vel_vec = T.reshape(final_vel, (final_vel.shape[0]*final_vel.shape[1],final_vel.shape[2]))
    initial_pos_vec = T.tile(positions, [final_pos.shape[0],1])
    initial_vel_vec = T.tile(initial_vel, [final_pos.shape[0],1])
    accept_vec, ndeltaH_vec = metropolis_hastings_accept(
        energy_prev = hamiltonian(initial_pos_vec, initial_vel_vec, energy_fn),
        energy_next = hamiltonian(final_pos_vec, final_vel_vec, energy_fn),
    )
    return accept_vec, initial_pos_vec, final_pos_vec, ndeltaH_vec, final_pos


