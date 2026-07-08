import numpy as np
import jax
import jax.numpy as jnp
from jax import hessian
import matplotlib.pyplot as plt


# we represent U, V as a single vector (vect(U).T, vect(V).T).T of shape (n * r + m * r,)
def vectorized_loss_fn(u_v, r, target_matrix, frozen_matrix, feature_matrix="eye"):
    n, m = target_matrix.shape
    u, v = u_v[: n * r], u_v[n * r :]
    U, V = u.reshape(n, r), v.reshape(m, r)
    if feature_matrix == "eye":
        feature_matrix = jnp.eye(m)
    return 0.5 * jnp.sum((U @ V.T @ feature_matrix - target_matrix + frozen_matrix) ** 2)


loss_hessian = hessian(vectorized_loss_fn, argnums=0)


def hessian_from_parameters(trainable_params, L, r, target_matrix, frozen_matrix, feature_matrix="eye"):
    n, m = target_matrix.shape
    if L != 1:
        raise ValueError("L must be 1 for Hessian computation")
    U = trainable_params["left"]["layer_0"]["weight"]
    V = trainable_params["right"]["layer_0"]["weight"].T
    u_v = jnp.concatenate([U, V], axis=0).flatten()
    return loss_hessian(u_v, r, target_matrix, frozen_matrix, feature_matrix)


def canonical_minimizer(target_matrix, r):
    U, S, V = jnp.linalg.svd(target_matrix, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    V = V[:r, :]
    U = U @ jnp.diag(jnp.sqrt(S))
    V = (jnp.diag(jnp.sqrt(S)) @ V).T
    return U, V


def condition_number(H, r):
    D, _ = jnp.linalg.eigh(H)
    return D[-1] / D[r**2]


def maximal_eigenvalue(H):
    D, _ = jnp.linalg.eigh(H)
    return D[-1]


def minimal_nonzero_eigenvalue(H, r):
    D, _ = jnp.linalg.eigh(H)
    return D[r**2]


def minimal_sharpness(target_matrix, r):
    U, V = canonical_minimizer(target_matrix, r)
    u_v = jnp.concatenate([U, V], axis=0).flatten()
    H = loss_hessian(u_v, r, target_matrix, np.zeros_like(target_matrix))
    D, _ = jnp.linalg.eigh(H)
    return D[-1]