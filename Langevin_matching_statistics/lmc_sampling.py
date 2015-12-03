# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:12:16 2015

@author: tian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 11:02:53 2015
perform the langevin dynamic sampling. Accept the updated states. 
Choose stepsize to be small and trajectory to be long enough
@author: tian
"""
import theano
import theano.tensor as T

s_rng = T.shared_randomstreams.RandomStreams(1234)

def langevin_move(initial_pos, initial_vel, decay_rates, stepsizes, n_steps, energy_fn):
    """
    Here, we perform n_steps langevin_dynamic, combined with partial momentum refreshment
    Iterate the following two steps: 
    (1) partial momentum refreshment: v(t) = alpha*v(t-1) + sqrt(1-alpha^2)*N(0,1)
    (2) Langevin simulation: p(t) = p(t-1) - 0.5*(epsilon^2)*dE/dp(t-1) + epsilon*p(t) 
    (1) & (2) are implemented in langevin_dynamic()
    
    initial_pos: the initial position. 
                 theano matrix: [n_sample] * [n_dim]
    initial_vel: the initial momentum.
                 theano matrix: [n_sample] * [n_dim]
    decay_rates: the constant alpha between [-1, 1] to update the momentums based on the previous one
                theano vector: [n_sample], each rate for each sample
    stepsizes:  the epsilon used to update the state position in langevin sampling
               theano vector: [n_sample], each stepsize for each sample
    n_steps: the number of iterations we want to perform to get the updated state
             theano scalar
    energy_fn: the underlying energy function we wish to sample from. 
    """
    def langevin_dynamic(pos, vel, decay, step):
        #decay_n_1 = decay.dimshuffle(0,'x')
        #noise = s_rng.normal(size = vel.shape)
        #vel_new = decay_n_1 * vel + T.sqrt(1.0 - decay_n_1**2) * noise
        # update position by half timestep
        stepsizes_n_1 = step.dimshuffle(0,'x')
        pos_half = pos + 0.5*stepsizes_n_1*vel
        # update velocity by full timestep
        vel_full = vel - stepsizes_n_1 * T.grad(energy_fn(pos_half).sum(), pos_half)
        # update position by half timestep
        pos_new = pos_half + 0.5*stepsizes_n_1 * vel_full        
        # get the random N(0,1) for each step
        noise = s_rng.normal(size = vel.shape)
        # partial momentum refreshment
        decay_n_1 = decay.dimshuffle(0,'x')
        vel_new = decay_n_1 * vel_full + T.sqrt(1.0 - decay_n_1**2) * noise    
        # DEBUG also return the noise
        return [pos_new, vel_new, noise], {}
    
    (all_pos, all_vel, all_noise), scan_updates = theano.scan(
        langevin_dynamic,
        outputs_info=[
            dict(initial=initial_pos),
            dict(initial=initial_vel),
            None
        ],
        non_sequences=[decay_rates, stepsizes],
        n_steps=n_steps)
        
   # final_pos = all_pos[-1]
    return all_pos, all_vel, all_noise, scan_updates

        
        
        
        
        
        
        
        

