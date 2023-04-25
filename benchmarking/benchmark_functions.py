import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm
from jax.scipy.special import erf
from jax import jit, grad, vmap, pmap

# General jax configuration
import jax
jax.config.update("jax_enable_x64", True)

from qiskit_dynamics.array import Array
Array.set_default_backend("jax")

import numpy as np
from time import time

from qiskit.quantum_info import Operator
from qiskit_dynamics import Solver, Signal
from qiskit_dynamics import DysonSolver, MagnusSolver

#######################################################
# Functions for running simulations and recording times
#######################################################

# rng seed for random input creation, necessary to keep benchmarks consistent
rng_seed = 123

def odeint_benchmark(
    tol, 
    evaluation_mode,
    num_inputs,
    num_processes=None,
    parallel_mode="vmap",
    save_state_file=None,
    compare_to_benchmark=None,
    benchmark_gradient=False,
):
    """Run benchmarks for odeint.
    
    Args:
        tol: Tolerance of simulation.
        evaluation_mode: Evaluation mode for the solver.
        num_inputs: Number of simulations to run.
        num_processes: Number of simulations to parallelize over at one time (if None, no 
            parallelization).
        parallel_mode: Either "pmap" or "vmap".
        save_state_file: File to save simulation states. If None, does not save.
        compare_to_benchmark: File to load benchmark solutions from and to compute
            distance. If None does not attempt.
        benchmark_gradient: Whether or not to run timings on gradients.
    """
    
    dim = 5

    # get model operators
    H0, Hdc, Hdt = build_hamiltonian(dim=dim)
    H0_B, Hdc_B, Hdt_B, v_t, fidelity = get_dressed_ops_and_fidelity_func(H0, Hdc, Hdt, dim=dim)

    # build solver
    solver = Solver(
        static_hamiltonian=H0_B,
        hamiltonian_operators=[Hdc_B, Hdt_B],
        rotating_frame=np.diag(H0_B),
        evaluation_mode=evaluation_mode
    )

    T = 200.0
    y0 = np.eye(dim ** 2, dtype=complex)

    def sim_function(params):
        signals = build_signals(params, v_t, T=T)

        results = solver.solve(
            t_span=[0, T],
            y0=y0,
            method="jax_odeint",
            atol=tol,
            rtol=tol,
            signals=signals
        )
        return results.y[-1]

    metrics = {
        "num_inputs": num_inputs, 
        "num_processes": num_processes,
        "parallel_mode": parallel_mode,
        "tol": tol,
        "evaluation_mode": evaluation_mode
    }

    ##########################
    # benchmark the simulation
    ##########################
    # generate random inputs
    rng = np.random.default_rng(rng_seed)
    input_params = jnp.array(rng.uniform(low=-2, high=2, size=(num_inputs, 6)))

    start = time()
    final_states = parallel_stitch(sim_function, input_params, num_processes, parallel_mode)
    sim_time = time() - start
    metrics["sim_time"] = sim_time

    if save_state_file is not None:
        final_states = np.asarray(final_states)
        np.save(save_state_file, final_states)
    
    if compare_to_benchmark is not None:
        benchmark_states = np.load(compare_to_benchmark)
        distances = np.array([distance(x, y, dim) for x, y in zip(benchmark_states, final_states)])
        metrics["average_distance"] = distances.sum() / num_inputs

    if not benchmark_gradient:
        return metrics

    ########################
    # benchmark the gradient
    ########################

    # construct function to grad
    grad_func = grad(lambda x: fidelity(sim_function(x)))
    
    start = time()
    grads = parallel_stitch(grad_func, input_params, num_processes, parallel_mode)
    grad_time = time() - start
    metrics["grad_time"] = grad_time

    return metrics


def perturbative_benchmark(
    expansion_method,
    chebyshev_order,
    expansion_order,
    n_steps,
    zero_freq=False,
    num_inputs=20,
    num_processes=None,
    parallel_mode="vmap",
    save_state_file=None,
    compare_to_benchmark=None,
    benchmark_gradient=False,
    mode="vmap",
):
    """Run benchmarks for perturbative solvers.
    
    Args:
        expansion_method: "magnus" or "dyson".
        chebyshev_order: Order of chebyshev approximation.
        expansion_order: Order of perturbative expansion.
        n_steps: Number of steps to break interval into.
        zero_freq: Whether or not to do approximation of envelope with carrier_freq set to target,
            or with carrier_freq set to 0 (approximate the full signal).
        num_inputs: Number of simulations to run.
        num_processes: Number of simulations to parallelize over at one time (if None, no 
            parallelization).
        parallel_mode: Either "pmap" or "vmap".
        save_state_file: File to save simulation states. If None, does not save.
        compare_to_benchmark: File to load benchmark solutions from and to compute
            distance. If None does not attempt.
        benchmark_gradient: Whether or not to run timings on gradients.
        mode: "vmap" or "scan".
    """

    dim = 5
    T = 200.0

    # get model operators
    H0, Hdc, Hdt = build_hamiltonian(dim=dim)
    H0_B, Hdc_B, Hdt_B, v_t, fidelity = get_dressed_ops_and_fidelity_func(H0, Hdc, Hdt, dim=dim)

    PerturbativeSolver = DysonSolver if expansion_method=="dyson" else MagnusSolver

    carrier_freqs = [v_t, v_t] if not zero_freq else [0., 0.]
    include_imag = [True, True] if not zero_freq else [False, False]

    start = time()
    solver = PerturbativeSolver(
        operators=[-1j * Hdc_B, -1j * Hdt_B],
        rotating_frame=-1j * np.diag(H0_B),
        dt=T / n_steps,
        carrier_freqs=carrier_freqs,
        include_imag=include_imag,
        chebyshev_orders=[chebyshev_order] * 2,
        expansion_order=expansion_order,
        #expansion_method=expansion_method,
        integration_method="jax_odeint",
        atol=1e-13,
        rtol=1e-13
    )
    construction_time = time() - start

    y0 = np.eye(dim ** 2, dtype=complex)

    def sim_function(params):
        signals = build_signals(params, v_t, T=T)

        results = solver.solve(
            t0=0.,
            n_steps=n_steps,
            y0=y0,
            signals=signals,
            mode=mode,
        )
        return results.y[-1]

    metrics = {
        "num_inputs": num_inputs, 
        "num_processes": num_processes,
        "parallel_mode": parallel_mode,
        "expansion_method": expansion_method,
        "chebyshev_order": chebyshev_order,
        "expansion_order": expansion_order,
        "n_steps": n_steps,
        "zero_freq": zero_freq,
        "construction_time": construction_time
    }

    ##########################
    # benchmark the simulation
    ##########################
    # generate random inputs
    rng = np.random.default_rng(rng_seed)
    input_params = np.array(rng.uniform(low=-2, high=2, size=(num_inputs, 6)))

    start = time()
    final_states = parallel_stitch(sim_function, input_params, num_processes, parallel_mode)
    sim_time = time() - start
    metrics["sim_time"] = sim_time

    if save_state_file is not None:
        final_states = np.asarray(final_states)
        np.save(save_state_file, final_states)

    if compare_to_benchmark is not None:
        benchmark_states = np.load(compare_to_benchmark)[0:num_inputs]
        distances = np.array([distance(x, y, dim) for x, y in zip(benchmark_states, final_states)])
        metrics["average_distance"] = distances.sum() / num_inputs

    if not benchmark_gradient:
        return metrics
    ########################
    # benchmark the gradient
    ########################

    # construct function to grad
    grad_func = grad(lambda x: fidelity(sim_function(x)))
    
    start = time()
    grads = parallel_stitch(grad_func, input_params, num_processes, parallel_mode)
    grad_time = time() - start
    metrics["grad_time"] = grad_time

    return metrics


def parallel_stitch(func, input_array, num_processes, parallel_mode):
    """Loop func over inputs in batches of vmap_num."""

    if num_processes is None:
        # if none, treat it as if it is set to vectorize over 1
        num_processes = 1
        parallel_func = jit(lambda x: jnp.array([func(x[0])]))
    else:
        parallel_func = pmap(func) if parallel_mode == "pmap" else jit(vmap(func))

    outcomes = []
    current_idx = 0

    while current_idx < len(input_array):
        last_sim_idx = min(current_idx + num_processes, len(input_array))
        outcomes.append(parallel_func(input_array[current_idx:last_sim_idx]))
        current_idx = last_sim_idx

    # combine outcomes
    out_array = outcomes[0]
    for out in outcomes[1:]:
        out_array = np.append(out_array, out, axis=0)
    
    return out_array


#################
# Model functions
#################

w_c_default = 2 * np.pi * 5.105
w_t_default = 2 * np.pi * 5.033
alpha_c_default = 2 * np.pi * (-0.33516)
alpha_t_default = 2 * np.pi * (-0.33721)
J_default = 2 * np.pi * 0.002

dim_default = 5

def build_hamiltonian(
    w_c=w_c_default,
    w_t=w_t_default,
    alpha_c=alpha_c_default,
    alpha_t=alpha_t_default,
    J=J_default,
    dim=dim_default
):
    """Build Hamiltonian operators from model parameters."""
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)
    ident2 = np.eye(dim ** 2)

    # operators on the control qubit (first tensor factor)
    a0 = np.kron(a, ident)
    adag0 = np.kron(adag, ident)
    N0 = np.kron(N, ident)

    # operators on the target qubit (first tensor factor)
    a1 = np.kron(ident, a)
    adag1 = np.kron(ident, adag)
    N1 = np.kron(ident, N)

    H0 = (
        w_c * N0
        + 0.5 * alpha_c * N0 @ (N0 - ident2)
        + w_t * N1
        + 0.5 * alpha_t * N1 @ (N1 - ident2)
        + J * (a0 @ adag1 + adag0 @ a1)
    )
    Hdc = 2 * np.pi * (a0 + adag0)
    Hdt = 2 * np.pi * (a1 + adag1)

    return H0, Hdc, Hdt


# Helper functions for organizing dressed basis
def basis_vec(ind, dimension):
    vec = np.zeros(dimension, dtype=complex)
    vec[ind] = 1.0
    return vec

def two_q_basis_vec(inda, indb, dimension):
    vec_a = basis_vec(inda, dimension)
    vec_b = basis_vec(indb, dimension)
    return np.kron(vec_a, vec_b)

def get_dressed_state_index(inda, indb, dimension, evector):
    b_vec = two_q_basis_vec(inda, indb, dimension)
    overlaps = np.abs(evector @ b_vec)
    return overlaps.argmax()

def get_dressed_state_and_energy(inda, indb, dimension, evals, evecs):
    ind = get_dressed_state_index(inda, indb, dimension, evecs)
    return evals[ind], evecs[ind]

def get_dressed_ops_and_fidelity_func(H0, Hdc, Hdt, dim=dim_default):
    """Convert operators into dressed basis of H0, the 0->1 transition frequency of the target,
    and and the fidelity function to ZX gate in the dressed basis.
    """

    # Diagonalize and get dressed energies/states for computational states.

    evals, B = jnp.linalg.eigh(H0)
    Badj = B.conj().transpose()

    # "target dressed frequency"
    E01, _ = get_dressed_state_and_energy(0, 1, dim, evals, B.transpose())
    v_t = E01 / (2 * np.pi)

    H0_B = Badj @ H0 @ B
    Hdc_B = Badj @ Hdc @ B
    Hdt_B = Badj @ Hdt @ B

    # Define fidelity with respect to the $Z \otimes X$ operator for the computational states.
    idx01 = get_dressed_state_index(0, 1, dim, B.transpose())
    idx10 = get_dressed_state_index(1, 0, dim, B.transpose())
    idx11 = get_dressed_state_index(1, 1, dim, B.transpose())

    e00 = np.zeros(dim ** 2, dtype=complex)
    e00[0] = 1.0
    e10 = np.zeros(dim ** 2, dtype=complex)
    e10[idx10] = 1.0
    e01 = np.zeros(dim ** 2, dtype=complex)
    e01[idx01] = 1.0
    e11 = np.zeros(dim ** 2, dtype=complex)
    e11[idx11] = 1.0

    # get target unitary
    S = np.array([e00, e01, e10, e11]).transpose()
    Sdag = S.conj().transpose()
    target = S @ jexpm(-1j * np.array(Operator.from_label("ZX")) * jnp.pi / 4) @ Sdag
    target_conj = target.conj()

    def fidelity(U):
        return jnp.abs(jnp.sum(target_conj * U)) ** 2 / (4 ** 2)
    
    return H0_B, Hdc_B, Hdt_B, v_t, fidelity

def distance(U, V, dim=dim_default):
    return jnp.linalg.norm(U - V) / dim

##########################
# signal parameterizations
##########################


def build_signals(params, v_t, sigma=7.0, risefall=2.0, T=200.0):
    cr_amp = params[0]
    rotary_amp = params[1]
    bipolar_amp = params[2]

    cr_phase = params[3]
    rotary_phase = params[4]
    bipolar_phase = params[5]

    cr_signal = Signal(
        lambda t: gaussian_square(t, cr_amp, sigma, risefall, T),
        carrier_freq=v_t,
        phase=cr_phase,
    )
    rotary_signal = Signal(
        lambda t: gaussian_square(t, rotary_amp, sigma, risefall, T),
        carrier_freq=v_t,
        phase=rotary_phase,
    )
    bipolar_signal = Signal(
        lambda t: bipolar_gaussian_square(t, bipolar_amp, sigma, risefall, T),
        carrier_freq=v_t,
        phase=bipolar_phase,
    )

    target_signal = (rotary_signal + bipolar_signal).flatten()

    return [cr_signal, target_signal]

# Define envelope functions
def gaussian_square(t, amp, sigma, risefall, T):
    """Gaussian square pulse."""

    t = Array(t).data
    C = jnp.exp(-((2 * risefall * sigma) ** 2) / (8 * sigma ** 2))
    den = (
        jnp.sqrt(jnp.pi * 2 * sigma ** 2)
        * erf(2 * risefall * sigma / (jnp.sqrt(8) * sigma))
        - 2 * risefall * sigma * C
    )
    return amp * jnp.piecewise(
        t,
        condlist=[t < (risefall * sigma), (T - t) < (risefall * sigma)],
        funclist=[
            lambda s: (
                jnp.exp(-((s - sigma * risefall) ** 2) / (2 * sigma ** 2)) - C
            )
            / den,
            lambda s: (
                jnp.exp(-((T - s - sigma * risefall) ** 2) / (2 * sigma ** 2)) - C
            )
            / den,
            lambda s: (1 - C) / den,
        ],
    )

def bipolar_gaussian_square(t, amp, sigma, risefall, T):
    t = Array(t).data
    unipolar = lambda s: gaussian_square(s, amp, sigma, risefall, T / 2)
    return jnp.piecewise(
        t,
        condlist=[t < (T / 2)],
        funclist=[unipolar, lambda s: -unipolar(s - T / 2)],
    )