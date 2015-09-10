# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 16:55:49 2015

@author: tian han
"""
"""HMC sampler for sampling_optimization process
   most of the function are from the theano tutorial hmc sampling page. we do the 
   modification for the MH acceptance step, cause we want use the differentiable 
   acceptance prob instead of just min()
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
    
def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    """
    Return final (position, velocity) obtained after an `n_steps` leapfrog
    updates, using Hamiltonian dynamics.

    Parameters
    ----------
    initial_pos: shared theano matrix
        Initial position at which to start the simulation
    initial_vel: shared theano matrix
        Initial velocity of particles
    stepsize: shared theano vector (update tian), it should contain n_samples elements, same as initial_pos.shape[0]
        Scalar value controlling amount by which to move, consider different stepsize for different particles
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.

    Returns
    -------
    rval1: theano matrix
        Final positions obtained after simulation
    rval2: theano matrix
        Final velocity obtained after simulation
    """

    def leapfrog(pos, vel, step):
        """
        Inside loop of Scan. Performs one step of leapfrog update, using
        Hamiltonian dynamics.

        Parameters
        ----------
        pos: theano matrix
            in leapfrog update equations, represents pos(t), position at time t
        vel: theano matrix
            in leapfrog update equations, represents vel(t - stepsize/2),
            velocity at time (t - stepsize/2)
        step: theano vector (update tian) it should have n_samples elements
            scalar value controlling amount by which to move, here, we want different step for different particles

        Returns
        -------
        rval1: [theano matrix, theano matrix]
            Symbolic theano matrices for new position pos(t + stepsize), and
            velocity vel(t + stepsize/2)
        rval2: dictionary
            Dictionary of updates for the Scan Op
        """
        # from pos(t) and vel(t-stepsize/2), compute vel(t+stepsize/2)
        dE_dpos = T.grad(energy_fn(pos).sum(), pos)
        #step_n_1: [n]*[1] matrix (makes a column out of the vector), in this way, theano will automatically broadcast it to elementwisely multiply
      
        step_n_1 = step.dimshuffle(0,'x')
        new_vel = vel - step_n_1 * dE_dpos
        # from vel(t+stepsize/2) compute pos(t+stepsize)
        new_pos = pos + step_n_1 * new_vel
        return [new_pos, new_vel], {}

    # compute velocity at time-step: t + stepsize/2
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    #stepsize_n_1: [n]*[1] matrix  (makes a column out of the vector). 
    stepsize_n_1 = stepsize.dimshuffle(0,'x')
    vel_half_step = initial_vel - 0.5 * stepsize_n_1 * dE_dpos

    # compute position at time-step: t + stepsize
    pos_full_step = initial_pos + stepsize_n_1 * vel_half_step

    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    (all_pos, all_vel), scan_updates = theano.scan(
        leapfrog,
        outputs_info=[
            dict(initial=pos_full_step),
            dict(initial=vel_half_step),
        ],
        non_sequences=[stepsize],
        n_steps=n_steps - 1)
    # final_pos and final_vel contains all the trajectories from length 2 to n_steps
    # 3D matrix: number of leapfrogsteps * number of samples * ndim
    #final_pos = all_pos[10:]
    #final_vel = all_vel[10:]
    final_pos = all_pos
    final_vel = all_vel
    final_vel_half = all_vel
    final_pos1 = all_pos[-1]
    final_vel1 = all_vel[-1]
    assert not scan_updates

    """
    deal with the single trajectory. (last trajectory)
    """
    energy_single = energy_fn(final_pos1)   
    final_vel1 = final_vel1 - 0.5 * stepsize_n_1 * T.grad(energy_single.sum(), final_pos1)
    
    # following code is used to deal with the all the trajectories. 
    """
    #energy is the 2D matrix: number of steps * number of samples
    energy, updates_energy = theano.scan(lambda final_s: energy_fn(final_s), sequences= final_pos)
    assert not updates_energy
    
    """
    #vectorize to compute energy
    """
    Since energy_fn() only accepts the 2D matrix, where each row is the sample, each column represents element in each dimen.
    Thus, we vectorize the 3D matrix to the 2D matrix in order to be fed into the energy_gn() function.
    
    final_pos_vec: reshape 3D matrix ([leapfrog_steps]*[samples]*[dims]) into 2D matrix ([steps*samples]*[dims]). 
    energy_vec: 1D vector [steps*samples], each element represents the energy for each sample.
    energy: reshape energy_vec back into the 2D matrix ([leapfrog_steps]*[samples])
    """
    final_pos_vec = T.reshape(final_pos, (final_pos.shape[0]*final_pos.shape[1],final_pos.shape[2]))
    energy_vec = energy_fn(final_pos_vec)
    energy = T.reshape(energy_vec,(final_pos.shape[0],final_pos.shape[1]))
       
    """
    Any better ways to avoid scan here? Here, we wanna get dE/ds. If we only have one trajectory, we can use something like
    T.grad(energy.sum(), position), while here, we have multiple trajectories, if we vectorize 3D energy to become 2D
    (#_of_steps * #_of_samples) * #_of_dims, then directly use T.grad(energy.sum(), position) will get questionable result,
    cause each trajectory is a symbolic function of the position.
    """
    # the following code is pretty tricky, if we do final_pos[i], then theano will consider it as another variable.
    """
    energy_grad: 3D matrix ([leapfrog_steps]*[samples]*[dims]), represents the d(E)/d(pos).
    """
    energy_grad, updates_energy_grad = theano.scan(lambda i, energy,final_pos: T.grad(energy[i].sum(), final_pos)[i], sequences=T.arange(final_pos.shape[0]), non_sequences=[energy, final_pos])
    assert not updates_energy_grad
    
    
    final_vel = final_vel - 0.5* stepsize_n_1 * energy_grad
    # return new proposal state
    return final_pos, final_vel, final_pos1, final_vel1, final_vel_half
    
def hmc_move(initial_vel,positions, energy_fn, stepsize, n_steps):
    """
    This function performs one-step of Hybrid Monte-Carlo sampling. We start by
    sampling a random velocity from a univariate Gaussian distribution, perform
    `n_steps` leap-frog updates using Hamiltonian dynamics and accept-reject
    using Metropolis-Hastings.

    Parameters
    ----------
    initial_vel: theano matrix
        Symbolic matrix whose row are velocity vectors. (Initialized each time when we do the optimization)
    positions: shared theano matrix
        Symbolic matrix whose rows are position vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.
    stepsize:  shared theano vector (update tian)
        Shared variable containing the stepsize to use for `n_steps` of HMC
        simulation steps. different values for different particles
    n_steps: integer
        Number of HMC steps to perform before proposing a new position.

    Returns
    -------
    rval1: theano matrix: [#_of_steps] * [#_of_samples]. accept prob. for all the trajectories
    rval2: theano vector: [#_of_samples]. For single trajectory.
    rval3: theano 3D tensor: [#_of_steps] * [#_of_samples] * [#_of_dims]. final pos. for all the trajectories.
    rval4: theano 2D matrix: [#_of_samples] * [#_of_dims]. final pos. for single trajectory. 
    """
    # end-snippet-1 start-snippet-2
    # sample random velocity
    #initial_vel = s_rng.normal(size=positions.shape)
    #initial_vel = theano.shared(value=numpy.zeros((3,3)))
    # end-snippet-2 start-snippet-3
    # perform simulation of particles subject to Hamiltonian dynamics
      
    # returned final_pos and final_vel are both 3D matrix which contain all possible trajectories from length 2 to n_steps
    """
    final_pos: 3D matrix ([leapfrog_steps]*[samples]*[dims], represents the final positions after doing one-step HMC with different leapfrog steps
    final_vel: 3D matrix ([leapfrog_steps]*[samples]*[dims], represents the final momentums after doing one-step HMC with different leapfrog steps
    final_pos1: 2D matrix ([samples]*[dims]), represents the final position after doing one-step HMC with one end leapfrog steps
    final_vel1: 2D matrix ([samples]*[dims]), represents the final momentum after doing one-step HMC with one end leapfrog steps
    """
    final_pos, final_vel, final_pos1, final_vel1, final_vel_half = simulate_dynamics(
        initial_pos=positions,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
   
    # end-snippet-3 start-snippet-4
    # accept/reject the proposed move based on the joint distribution
    accept1, ndeltaH1 = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos1, final_vel1, energy_fn),
    )
    
    """
    # accept is the 2D matrix: number of steps * number of samples
    accept, updates_accept = theano.scan(lambda final_s, final_v: metropolis_hastings_accept(
                                                                  energy_prev=hamiltonian(positions, initial_vel, energy_fn),
                                                                  energy_next=hamiltonian(final_s, final_v, energy_fn)),
                                                                  sequences=[final_pos, final_vel])
    assert not updates_accept
    """
    
    
    #vectorize first to compute the accept pro. and then reshape to 2D. 
    """
    Since the metropolis_hasting_accept() only accept the 2D matrix, so here we first reshape the final_pos and final_vel
    to become 2D, then reshape them back to 3D. 
    
    final_pos_vec: 2D matrix ([leapfrog_steps*samples]*[dims])
    final_vel_vec: 2D matrix ([leapfrog_steps*samples]*[dims])
    initial_pos_vec: 2D matrix. ([leapfrog_steps*samples]*[dims]). previously, positions ([samples]*[dims]) is only related to
                     one leapfrog step, so here we tile positions into [leapfrog_steps]*[1], so we get (leapfrog_steps) copies 
                     of 2D positions matrix. 
    initial_vel_vec: 2D matrix. ([leapfrog_steps*samples]*[dims]). previously, initial_vel ([samples]*[dims]) is only related to
                     one leapfrog step, so here we tile initial_vel into [leapfrog_steps]*[1], so we get (leapfrog_steps) copies 
                     of 2D initial_vel matrix. 
    accept_vec: 1D vector([leapfrog_steps*samples]) represents the accept prob. for each sample.
    accept: 2D matrix, ([leapfrog_steps]*[samples]).
    """
    final_pos_vec = T.reshape(final_pos, (final_pos.shape[0]*final_pos.shape[1],final_pos.shape[2]))
    final_vel_vec = T.reshape(final_vel, (final_vel.shape[0]*final_vel.shape[1],final_vel.shape[2]))
    initial_pos_vec = T.tile(positions, [final_pos.shape[0],1])
    initial_vel_vec = T.tile(initial_vel, [final_pos.shape[0],1])
    accept_vec, ndeltaH_vec = metropolis_hastings_accept(
        energy_prev = hamiltonian(initial_pos_vec, initial_vel_vec, energy_fn),
        energy_next = hamiltonian(final_pos_vec, final_vel_vec, energy_fn),
    )
    accept = T.reshape(accept_vec, [final_pos.shape[0], final_pos.shape[1]])
    ndeltaH = T.reshape(ndeltaH_vec, [final_pos.shape[0], final_pos.shape[1]])
    
    # end-snippet-4
    return  accept, accept1,final_pos, final_pos1, ndeltaH

    

