# %% [markdown]
# # Perturbative solver demo
#
# This demo walks through the construction and usage of `PerturbativeSolver` objects for simulating a 2 transmon gate, comparing to traditional solvers using both dense and sparse arrays.

# %%
from time import time
import argparse

import numpy as np
import pandas as pd
import logging

import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm
from jax.scipy.special import erf
from jax import jit, value_and_grad, vmap


from qiskit.quantum_info import Operator

from qiskit_dynamics import Solver, Signal
from qiskit_dynamics.perturbation import PerturbativeSolver

import sqlite3
from sqlite3 import Error

#%%
def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB succesful")
    except Error as e:
        print(f"the error '{e}' occured")
    return connection


# %% [markdown]
# Configure to use JAX.

# %%
from qiskit_dynamics.array import Array

# configure jax to use 64 bit mode
import jax

jax.config.update("jax_enable_x64", True)


def main_runner(cpu_count, output_file, vmap_flag, arg_solv="all", num_inputs=10):
# cpu_count=36
# output_file='notebook'
# vmap_flag=True
# arg_solv='dyson'
    results_db_path = (
        "/u/brosand/danDynamics/multivariable_dyson_magnus/solver_results.sqlite"
    )

    connection = create_connection(results_db_path)
    cursor = connection.cursor()
    # cursor.execute(SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}')
    if (
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='benchmarks'"
        ).fetchall()[0][0]
        == 0
    ):
        cursor.execute(
            "CREATE TABLE benchmarks (solver TEXT, jit_time FLOAT, ave_run_time FLOAT, ave_distance FLOAT, jit_grad_time FLOAT, jit_vmap_time FLOAT, ave_grad_run_time FLOAT, construction_time FLOAT, step_count FLOAT, tol FLOAT, cpus INTEGER, gpus INTEGER, cheb_order INTEGER, exp_order INTEGER, vmap INTEGER)"
        )
    def write_result(
        cursor,
        solver,
        jit_time,
        ave_run_time,
        ave_distance,
        jit_grad_time,
        ave_grad_run_time,
        cpus,
        gpus,
        jit_vmap_time=0,
        vmap=False,
        tol=0,
        construction_time=0,
        step_count=0,
        exp_order=0,
        cheb_order=10,
    ):
        if vmap:
            vmap = 1
        else:
            vmap = 0

        # columns = ["solver", "jit_time", "ave_run_time", "ave_distance", "jit_grad", "ave_grad_run_time", "construction_time", "step_count", "tol", "cpus", "gpus", "cheb_order", "exp_order", "vmap"]
        columns = [solver, jit_time, ave_run_time, ave_distance, jit_grad_time, jit_vmap_time, ave_grad_run_time, construction_time, step_count, tol, cpus, gpus, cheb_order, exp_order, vmap]
        columns = [f'"{item}"' if isinstance(item, str) else str(item) for item in columns]
        
        string_columns=",".join(columns)
        cursor.execute(
            f"INSERT INTO benchmarks VALUES ({string_columns})"

        )
            # f"INSERT INTO benchmarks(solver {solver}, jit_time {jit_time}, ave_run_time {ave_run_time}, ave_distance {ave_distance}, jit_grad_time {jit_grad_time}, ave_grad_run_time {ave_grad_run_time}, construction_time {construction_time}, step_count {step_count}, tol {tol}, cpus {cpus}, gpus {gpus}, cheb_order {cheb_order}, exp_order {exp_order}, vmap {vmap})"
        connection.commit()
        logging.warning("committed to sql table")



    # tell JAX we are using CPU
    gpu = False
    if cpu_count == 0:
        gpu = True
    if gpu:
        jax.config.update("jax_platform_name", "gpu")
        logging.warning("Using GPU")
    else:
        jax.config.update("jax_platform_name", "cpu")
        logging.warning("Using CPU")

    # set default backend
    Array.set_default_backend("jax")

    # %% [markdown]
    # # 1. Define envelope functions
    #
    # We define a Gaussian square and bipolar Gaussian square pulse shape.

    # %%
    def gaussian_square(t, amp, sigma, risefall, T):
        """Gaussian square pulse."""

        t = Array(t).data
        C = jnp.exp(-((2 * risefall * sigma) ** 2) / (8 * sigma**2))
        den = (
            jnp.sqrt(jnp.pi * 2 * sigma**2)
            * erf(2 * risefall * sigma / (jnp.sqrt(8) * sigma))
            - 2 * risefall * sigma * C
        )
        return amp * jnp.piecewise(
            t,
            condlist=[t < (risefall * sigma), (T - t) < (risefall * sigma)],
            funclist=[
                lambda s: (
                    jnp.exp(-((s - sigma * risefall) ** 2) / (2 * sigma**2)) - C
                )
                / den,
                lambda s: (
                    jnp.exp(-((T - s - sigma * risefall) ** 2) / (2 * sigma**2)) - C
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

    # %% [markdown]
    # Plot an example.

    # %%
    T = 200.0
    risefall = 2.0
    sigma = 7.0
    amp = 4.0

    test = jnp.vectorize(lambda t: bipolar_gaussian_square(t, amp, sigma, risefall, T))

    sig = Signal(test)

    # %%
    sig.draw(0, T, 1000, function="envelope")

    # %% [markdown]
    # # 2. Construct model operators
    #
    # We construct a two transmon model:
    #
    # $$H(t) = 2 \pi \nu a_0 a_0^\dagger + 2 \pi r (a_0 + a_0^\dagger) \\
    #    + 2 \pi \nu a_1 a_1^\dagger + 2 \pi r (a_1 + a_1^\dagger)\\
    #    + 2 \pi J (a_0a_1^\dagger + a_0^\dagger a_1)$$

    # %%
    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    alpha_c = 2 * np.pi * (-0.33516)
    alpha_t = 2 * np.pi * (-0.33721)
    J = 2 * np.pi * 0.002

    dim = 5

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)
    ident2 = np.eye(dim**2)

    # operators on the control qubit (first tensor factor)
    a0 = np.kron(a, ident)
    adag0 = np.kron(adag, ident)
    N0 = np.kron(N, ident)

    # operators on the target qubit (first tensor factor)
    a1 = np.kron(ident, a)
    adag1 = np.kron(ident, adag)
    N1 = np.kron(ident, N)

    # %%
    H0 = (
        w_c * N0
        + 0.5 * alpha_c * N0 @ (N0 - ident2)
        + w_t * N1
        + 0.5 * alpha_t * N1 @ (N1 - ident2)
        + J * (a0 @ adag1 + adag0 @ a1)
    )
    Hdc = 2 * np.pi * (a0 + adag0)
    Hdt = 2 * np.pi * (a1 + adag1)

    # %% [markdown]
    # ## 2.1 Get the dressed computational states qubit frequencies

    # %%
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

    def get_dressed_state_and_energy(inda, indb, dimension, evecs):
        ind = get_dressed_state_index(inda, indb, dimension, evecs)
        return evals[ind], evecs[ind]

    # %% [markdown]
    # Diagonalize and get dressed energies/states for computational states.

    # %%
    evals, B = jnp.linalg.eigh(H0)
    Badj = B.conj().transpose()

    E00, dressed00 = get_dressed_state_and_energy(0, 0, dim, B.transpose())
    E01, dressed01 = get_dressed_state_and_energy(0, 1, dim, B.transpose())
    E10, dressed10 = get_dressed_state_and_energy(1, 0, dim, B.transpose())
    E11, dressed11 = get_dressed_state_and_energy(1, 1, dim, B.transpose())

    # "target dressed frequency"
    v_t = E01 / (2 * np.pi)

    # %%
    H0_B = Badj @ H0 @ B
    Hdc_B = Badj @ Hdc @ B
    Hdt_B = Badj @ Hdt @ B

    # %% [markdown]
    # Define fidelity with respect to the $Z \otimes X$ operator for the computational states.

    # %%
    idx00 = 0
    idx01 = get_dressed_state_index(0, 1, dim, B.transpose())
    idx10 = get_dressed_state_index(1, 0, dim, B.transpose())
    idx11 = get_dressed_state_index(1, 1, dim, B.transpose())

    e00 = np.zeros(dim**2, dtype=complex)
    e00[0] = 1.0
    e10 = np.zeros(dim**2, dtype=complex)
    e10[idx10] = 1.0
    e01 = np.zeros(dim**2, dtype=complex)
    e01[idx01] = 1.0
    e11 = np.zeros(dim**2, dtype=complex)
    e11[idx11] = 1.0

    # set up observables
    S = np.array([e00, e01, e10, e11]).transpose()
    Sdag = S.conj().transpose()

    ZX = S @ np.array(Operator.from_label("ZX")) @ Sdag

    target = S @ jexpm(-1j * np.array(Operator.from_label("ZX")) * jnp.pi / 4) @ Sdag
    target_conj = target.conj()

    def fidelity(U):
        return jnp.abs(jnp.sum(target_conj * U)) ** 2 / (4**2)

    # %% [markdown]
    # # 3. Construct dense version of simulation
    #
    # Here we construct a function for simulating the system in the rotating frame of the drift, using a standard ODE solver, and dense arrays.

    # %%
    dense_solver = Solver(
        static_hamiltonian=H0_B,
        hamiltonian_operators=[Hdc_B, Hdt_B],
        rotating_frame=np.diag(H0_B),
    )

    y0 = np.eye(dim**2, dtype=complex)

    def ode_sim(params, tol):
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

        solver_copy = dense_solver.copy()

        solver_copy.signals = [cr_signal, target_signal]
        results = solver_copy.solve(
            t_span=[0, T], y0=y0, method="jax_odeint", atol=tol, rtol=tol
        )
        return results.y[-1]

    def ode_obj(params, tol):
        return fidelity(ode_sim(params, tol))

    # %% [markdown]
    # ## Setup a collection of inputs values and create benchmark final unitaries

    # %%
    rng = np.random.default_rng(123)

    # num_inputs=30
    input_params = jnp.array(rng.uniform(low=-2, high=2, size=(num_inputs, 6)))

    # orig = jnp.array([1.4, 1., 0.3, 0., 0., 0.])

    # %%
    benchmark_sim = jit(lambda x: ode_sim(x, 1e-14))

    benchmark_yfs = [benchmark_sim(x) for x in input_params]

    # %% [markdown]
    # ## Create error metrics and function for running sims

    # %%
    def distance(U, V):
        return jnp.linalg.norm(U - V) / dim

    target = S @ jexpm(-1j * np.array(Operator.from_label("ZX")) * jnp.pi / 4) @ Sdag
    target_conj = target.conj()

    def gate_fidelity(U):
        return jnp.abs(jnp.sum(target_conj * U)) ** 2 / (4**2)

    # %%
    from time import time

    def compute_solver_metrics(sim_func):
        sim_func = jit(sim_func)

        # time to jit
        start = time()
        sim_func(input_params[0]).block_until_ready()
        jit_time = time() - start

        # loop over and run simulations
        vmap_sim_func = jit(vmap(sim_func))
        start = time()
        vmap_sim_func(jnp.array([input_params[0]])).block_until_ready()
        vmap_jit_time = time() - start

        start = time()
        if vmap_flag:
            # for i in range(num_inputs)[::2]:
            #     short_input=jnp.array([input_params[i], input_params[i+1]])
                # yfs = vmap_sim_func(short_input).block_until_ready()
            yfs = vmap_sim_func(input_params).block_until_ready()
        else:
            yfs = [sim_func(x).block_until_ready() for x in input_params]
        ave_run_time = (time() - start) / len(input_params)

        distances = []
        for yf, benchmark_yf in zip(yfs, benchmark_yfs):
            distances.append(distance(yf, benchmark_yf))

        ave_distance = np.sum(distances).real / len(input_params)

        def fid_func(x):
            yf = sim_func(x)
            return gate_fidelity(yf)

        jit_grad_fid_func = jit(value_and_grad(fid_func))

        # time to jit
        start = time()
        jit_grad_fid_func(input_params[0])[0].block_until_ready()
        jit_grad_time = time() - start

        # time to compute gradients
        start = time()
        for x in input_params:
            jit_grad_fid_func(x)[0].block_until_ready()
        ave_grad_run_time = (time() - start) / len(input_params)

        return {
            "jit_time": jit_time,
            "ave_run_time": ave_run_time,
            "ave_distance": ave_distance,
            "jit_grad_time": jit_grad_time,
            "ave_grad_run_time": ave_grad_run_time,
            "jit_vmap_time": vmap_jit_time,
        }

    # %% [markdown]
    # # Dense simulation
    #
    # Run the sims for dense simulation at various tolerances.
    #
    # we should run this for up to `k==1e-13`, and possibly even for intermediate values to fill out the curve.

    # %%
    tols = [10**-k for k in range(6, 15)]

    if cpu_count == 0:
        gpu_count = 1
    else:
        gpu_count = 0
    if arg_solv in ["dense", "all"]:
        dense_results = []
        for tol in tols:
            metrics = compute_solver_metrics(lambda params: ode_sim(params, tol))
            dense_results.append(metrics)

            write_result(
                cursor=cursor,
                solver="dense",
                jit_time=metrics["jit_time"],
                ave_run_time=metrics["ave_run_time"],
                ave_distance=metrics["ave_distance"],
                jit_grad_time=metrics["jit_grad_time"],
                ave_grad_run_time=metrics["ave_grad_run_time"],
                jit_vmap_time=metrics['jit_vmap_time'],
                cpus=cpu_count,
                gpus=gpu_count,
                vmap=vmap_flag,
                tol=0,
                construction_time=0,
                step_count=0,
                exp_order=0,
                cheb_order=10,
            )

        dense_results_df = pd.DataFrame(dense_results)
        dense_results_df["tol"] = tols
        # dense_results_df.to_csv('dense_results_cpu_{}.csv'.format(cpu_count))
        dense_results_df.to_csv(
            f"/u/brosand/danDynamics/multivariable_dyson_magnus/results/dense_results_{output_file}.csv"
        )
    # %% [markdown]
    # # Sparse version of simulation
    #
    # For sparse simulation we need to make sure we are in a basis in which the operators are actually sparse.

    # %%
    sparse_solver = Solver(
        static_hamiltonian=H0,
        hamiltonian_operators=[Hdc, Hdt],
        rotating_frame=np.diag(H0),
        evaluation_mode="sparse",
    )

    y0_sparse = B @ y0

    def ode_sparse_sim(params, tol):
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

        solver_copy = sparse_solver.copy()

        solver_copy.signals = [cr_signal, target_signal]
        results = solver_copy.solve(
            t_span=[0, T], y0=y0_sparse, method="jax_odeint", atol=tol, rtol=tol
        )

        # transfer unitary into same basis and frame as the dense simulation
        U = Array(Badj) @ solver_copy.model.rotating_frame.state_out_of_frame(
            T, results.y[-1]
        )
        U = dense_solver.model.rotating_frame.state_into_frame(T, U).data

        return U

    # %%
    if arg_solv in ["sparse", "all"]:
        sparse_results = []
        for tol in tols:
            metrics = compute_solver_metrics(lambda params: ode_sparse_sim(params, tol))
            sparse_results.append(metrics)
            write_result(
                cursor=cursor,
                solver="sparse",
                jit_time=metrics["jit_time"],
                ave_run_time=metrics["ave_run_time"],
                ave_distance=metrics["ave_distance"],
                jit_grad_time=metrics["jit_grad_time"],
                ave_grad_run_time=metrics["ave_grad_run_time"],
                jit_vmap_time=metrics['jit_vmap_time'],
                cpus=cpu_count,
                gpus=gpu_count,
                vmap=vmap_flag,
                tol=0,
                construction_time=0,
                step_count=0,
                exp_order=0,
                cheb_order=10,
            )

        sparse_results_df = pd.DataFrame(sparse_results)
        sparse_results_df["tol"] = tols
        # sparse_results_df.to_csv('sparse_results_cpu_{}.csv'.format(cpu_count))
        sparse_results_df.to_csv(
            f"/u/brosand/danDynamics/multivariable_dyson_magnus/results/sparse_results_{output_file}.csv"
        )
    # %% [markdown]
    # # Dyson solver

    # %%
    # system information
    operators = [-1j * Hdc_B, -1j * Hdt_B]
    carrier_freqs = [v_t, v_t]
    frame_operator = -1j * np.diag(H0_B)

    def perturbative_solver_metrics(
        n_steps, expansion_order, chebyshev_order, expansion_method="dyson"
    ):
        dt = T / n_steps

        # construct solver
        start = time()
        perturb_solver = PerturbativeSolver(
            operators=operators,
            rotating_frame=frame_operator,
            dt=dt,
            carrier_freqs=carrier_freqs,
            chebyshev_orders=[chebyshev_order] * 2,
            expansion_method=expansion_method,
            expansion_order=expansion_order,
            integration_method="jax_odeint",
            atol=1e-13,
            rtol=1e-13,
        )
        construction_time = time() - start

        def perturb_sim(params):
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

            return perturb_solver.solve([cr_signal, target_signal], y0, 0.0, n_steps)

        results = compute_solver_metrics(perturb_sim)
        results["construction_time"] = construction_time
        return results

    # %% [markdown]
    # For reference, this is a value that gives a very high quality approximation.

    # %%
    import warnings

    step_counts = [10000, 20000, 30000, 40000, 50000]
    exp_orders = [2, 3, 4, 5]
    cheb_orders = [0, 1, 2]

    if arg_solv in ["dyson", "all"]:
        perturbative_results = []
        logging.warning("HIII")
        for step_count in step_counts:
            for exp_order in exp_orders:
                for cheb_order in cheb_orders:
                    warnings.warn(
                        f"step: {step_count}, exp: {exp_order}, cheb: {cheb_order}"
                    )
                    test = perturbative_solver_metrics(
                        n_steps=step_count,
                        expansion_order=exp_order,
                        chebyshev_order=cheb_order,
                        expansion_method="dyson",
                    )
                    test["step_count"] = step_count
                    test["exp_order"] = exp_order
                    test["cheb_order"] = cheb_order
                    metrics = test
                    write_result(
                        cursor=cursor,
                        solver="dyson",
                        jit_time=metrics["jit_time"],
                        ave_run_time=metrics["ave_run_time"],
                        ave_distance=metrics["ave_distance"],
                        jit_grad_time=metrics["jit_grad_time"],
                        ave_grad_run_time=metrics["ave_grad_run_time"],
                        jit_vmap_time=metrics['jit_vmap_time'],
                        cpus=cpu_count,
                        gpus=gpu_count,
                        vmap=vmap_flag,
                        tol=0,
                        construction_time=metrics["construction_time"],
                        step_count=metrics["step_count"],
                        exp_order=metrics["exp_order"],
                        cheb_order=metrics["cheb_order"],
                    )
                    perturbative_results.append(test)

                    pert_df = pd.DataFrame(perturbative_results)
                    # pert_df.to_csv('/u/brosand/danDynamics/multivariable_dyson_magnus/results/perturbative_results_cpu_dyson_{}.csv'.format(cpu_count))
                    pert_df.to_csv(
                        f"/u/brosand/danDynamics/multivariable_dyson_magnus/results/dyson_results_{output_file}.csv"
                    )

    if arg_solv in ["magnus", "all"]:
        perturbative_results = []

        for step_count in step_counts:
            for exp_order in exp_orders:
                for cheb_order in cheb_orders:
                    warnings.warn(
                        f"step: {step_count}, exp: {exp_order}, cheb: {cheb_order}"
                    )
                    test = perturbative_solver_metrics(
                        n_steps=step_count,
                        expansion_order=exp_order,
                        chebyshev_order=cheb_order,
                        expansion_method="magnus",
                    )
                    test["step_count"] = step_count
                    test["exp_order"] = exp_order
                    test["cheb_order"] = cheb_order
                    metrics = test

                    write_result(
                        cursor=cursor,
                        solver="magnus",
                        jit_time=metrics["jit_time"],
                        ave_run_time=metrics["ave_run_time"],
                        ave_distance=metrics["ave_distance"],
                        jit_grad_time=metrics["jit_grad_time"],
                        ave_grad_run_time=metrics["ave_grad_run_time"],
                        jit_vmap_time=metrics['jit_vmap_time'],
                        cpus=cpu_count,
                        gpus=gpu_count,
                        vmap=vmap_flag,
                        tol=0,
                        construction_time=metrics["construction_time"],
                        step_count=metrics["step_count"],
                        exp_order=metrics["exp_order"],
                        cheb_order=metrics["cheb_order"],
                    )

                    perturbative_results.append(test)

                    pert_df = pd.DataFrame(perturbative_results)
                    # pert_df.to_csv('/u/brosand/danDynamics/multivariable_dyson_magnus/results/perturbative_results_cpu_magnus_{}.csv'.format(cpu_count))
                    pert_df.to_csv(
                        f"/u/brosand/danDynamics/multivariable_dyson_magnus/results/magnus_results_{output_file}.csv"
                    )

# %% [markdown]
# We should generate data for perturbative solvers, for both Dyson and Magnus, treating `n_steps` analogously to `tol` for the usual solvers. I.e. for different expansion orders and chebyshev orders, generate the metrics for a range of `n_steps`. We'll need to play around to see what actual values of `n_steps` to explore (e.g. cranking it really high to get as high a tolerance as possible).
#
# For CPU, can we pin this to a single core?

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 2q sim.")
    parser.add_argument("--solver", default="all")
    parser.add_argument("--output_name")
    parser.add_argument("--cpus", default=8, type=int)
    parser.add_argument("--n_inputs", default=8, type=int)
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--vmap", dest="vmap", action="store_true")
    parser.add_argument("--norft", dest="rft", action="store_false")
    parser.set_defaults(rft=True)
    parser.set_defaults(vmap=False)
    parser.set_defaults(test=False)

    args = parser.parse_args()

    logging.warning(f"arg_solv is {args.solver}")

    main_runner(
        cpu_count=args.cpus,
        output_file=args.output_name,
        arg_solv=args.solver,
        vmap_flag=args.vmap,
        num_inputs=args.n_inputs,
        # rft=args.rft,
        # args_dict=vars(args),
    )