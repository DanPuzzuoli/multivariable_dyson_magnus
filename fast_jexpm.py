# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This module contains modified code from the JAX library, for the purposes
# of specialization to the applications in this repository. The JAX
# repository is available at:
# https://github.com/google/jax

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import scipy.linalg
import textwrap

from jax import jit, vmap
from jax import api
from jax import lax
from jax._src.numpy.util import _wraps
import jax.numpy as jnp

_expm_description = textwrap.dedent("""
This is a modified version of jax.scipy.linalg.expm. Original source
available at: https://github.com/google/jax/blob/master/jax/_src/scipy/linalg.py.
This version has been modified to always compute the highest-order Pade
approximant. While in principle this should slow things down, in practice
(in our use cases) it seems to decrease both compilation and execution times,
as it avoids several conditionals and partial computations that go into
deciding which order to use.
""")


@_wraps(scipy.linalg.expm, lax_description=_expm_description)
def expm(A, *, max_squarings=16):
    return _expm(A, max_squarings)

@partial(jit, static_argnums=(1,))
def _expm(A, max_squarings):
    P, Q, n_squarings = _calc_P_Q(A)

    def _nan(args):
        A, *_ = args
        return jnp.full_like(A, jnp.nan)

    def _compute(args):
        A, P, Q = args
        R = _solve_P_Q(P, Q)
        R = _squaring(R, n_squarings)
        return R

    R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q))
    return R

@jit
def _calc_P_Q(A):
    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    A_L1 = jnp.linalg.norm(A, 1)
    n_squarings = 0
    if A.dtype == 'float64' or A.dtype == 'complex128':
        maxnorm = 5.371920351148152
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A = A / 2**n_squarings
        U, V = _pade13(A)
    elif A.dtype == 'float32' or A.dtype == 'complex64':
        maxnorm = 3.925724783138660
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A = A / 2**n_squarings
        U,V = _pade7(A)
    else:
        raise TypeError("A.dtype={} is not supported.".format(A.dtype))
    P = U + V  # p_m(A) : numerator
    Q = -U + V # q_m(A) : denominator
    return P, Q, n_squarings

def _solve_P_Q(P, Q):
    return jnp.linalg.solve(Q, P)

def _precise_dot(A, B):
    return jnp.dot(A, B, precision=lax.Precision.HIGHEST)

@jit
def _squaring(R, n_squarings):
    # squaring step to undo scaling
    def _squaring_precise(x):
        return _precise_dot(x, x)

    def _identity(x):
        return x

    def _scan_f(c, i):
        return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None
    res, _ = lax.scan(_scan_f, R, jnp.arange(16))

    return res

def _pade7(A):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    A6 = _precise_dot(A4, A2)
    U = _precise_dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade13(A):
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600., 670442572800.,
         33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    A6 = _precise_dot(A4, A2)
    U = _precise_dot(A, _precise_dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = _precise_dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V
