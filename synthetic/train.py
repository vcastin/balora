import jax
import jax.numpy as jnp
import numpy as np
import functools
import copy
from layers import (
    merge_params,
    apply_linear,
    apply_lora_network,
    init_lora_network,
    get_trainable_parameters_linear,
    get_trainable_parameters_lora,
)

# def has_nan(tree):
#         return jnp.any(jnp.concatenate([jnp.ravel(jnp.isnan(x)) for x in jax.tree_util.tree_leaves(tree)]))


def newton_schulz(A, num_iters=10):
    X = A
    for _ in range(num_iters):
        X = 0.5 * X @ (3 * jnp.eye(A.shape[0]) - X.T @ X)
    return X


def polar_decomposition(A, side):
    """Computes the polar decomposition of a matrix A."""
    U, S, V = jnp.linalg.svd(A, full_matrices=False)
    if side == "RS":
        return {"S": V.T @ jnp.diag(S) @ V, "R": U @ V}
    if side == "SR":
        return {"S": U @ jnp.diag(S) @ U.T, "R": U @ V}

# def polar_decomposition(A):
#     """Computes the polar decomposition of a matrix A."""
#     n, m = A.shape
#     if n > m:
#         output = polar_decomposition(A.T)
#         output["R"] = output["R"].T
#         output["side"] = "RS"
#         return output
#     U, S, V = jnp.linalg.svd(A, full_matrices=False)
#     return {"S": U @ jnp.diag(S) @ U.T, "R": U @ V, "side": "SR"}


def project_balanced(trainable_params, mode="old"):
    projected_params = copy.deepcopy(trainable_params)
    for key in trainable_params["left"]:
        A = trainable_params["left"][key]["weight"]
        B = trainable_params["right"][key]["weight"]
        r = A.shape[1]
        if mode == "old":
            U, S, V = jnp.linalg.svd(A @ B, full_matrices=False)
            A_proj = (U * jnp.sqrt(S))[:, :r]
            B_proj = (jnp.sqrt(S)[:, None] * V)[:r, :]
        elif mode == "new":
            polar_A = polar_decomposition(A, side="RS")
            polar_B = polar_decomposition(B, side="SR")
            S = (polar_A["S"] + polar_B["S"]) * 0.5
            A_proj = polar_A["R"] @ S
            B_proj = S @ polar_B["R"]
        elif mode == "best":
            U, S, V = jnp.linalg.svd(A @ B, full_matrices=False)
            U = U[:, :r]
            V = V[:r, :]
            S = S[:r]
            A_proj = U * jnp.sqrt(S)
            B_proj = jnp.sqrt(S)[:, None] * V
            to_approx = 0.5 * jnp.sqrt(S)[:, None] * (U.T @ A + V @ B.T)
            to_approx /= jnp.linalg.norm(to_approx)
            O = newton_schulz(to_approx)
            A_proj = (A_proj @ O)
            B_proj = (O.T @ B_proj)
        projected_params["left"][key]["weight"] = A_proj
        projected_params["right"][key]["weight"] = B_proj
    return projected_params


def solve_lyapunov(Q, M):
    d, U = jnp.linalg.eigh(Q)
    M_tilde = U.T @ M @ U
    eps = 1e-12
    S_tilde = M_tilde / (d[:, None] + d[None, :] + eps)
    S = U @ S_tilde @ U.T
    return 0.5 * (S + S.T)


def project_gradient(trainable_params, grad):
    correction_params = copy.deepcopy(grad)
    for key in trainable_params["left"]:
        A = trainable_params["left"][key]["weight"]
        B = trainable_params["right"][key]["weight"]
        Q = A.T @ A  # WARNING: we need A.T @ A == B @ B.T
        # jax.debug.print("A^T A == B B^T? {}", jnp.allclose(A.T @ A, B @ B.T))
        # Compute the RHS term of the Lyapunov equation
        gradA = grad["left"][key]["weight"]
        gradB = grad["right"][key]["weight"]
        M = 0.5 * (A.T @ gradA + gradA.T @ A - B @ gradB.T - gradB @ B.T)
        # Solve Q S + S Q = M
        S = solve_lyapunov(Q, M)
        # Project to tangent space
        correction_params["left"][key]["weight"] = gradA - A @ S
        correction_params["right"][key]["weight"] = gradB + S @ B
    return correction_params


def retraction(updated_trainable_params, trainable_params):
    retracted_params = copy.deepcopy(trainable_params)
    for key in trainable_params["left"]:
        A = trainable_params["left"][key]["weight"]
        B = trainable_params["right"][key]["weight"]
        Q = A.T @ A  # WARNING: we need A.T @ A == B @ B.T
        # jax.debug.print("A^T A == B B^T? {}", jnp.allclose(A.T @ A, B @ B.T))
        # Compute Q^(1/2) and Q^(-1/2)
        eigvals, eigvecs = jnp.linalg.eigh(Q)
        sqrtQ = (eigvecs * jnp.sqrt(eigvals)) @ eigvecs.T
        sqrtinvQ = (eigvecs * (1.0 / jnp.sqrt(eigvals))) @ eigvecs.T
        # Compute polar decomposition (via SVD) of the updated matrices
        Aup = updated_trainable_params["left"][key]["weight"]
        Bup = updated_trainable_params["right"][key]["weight"]
        Atilde = Aup @ sqrtinvQ
        Btilde = sqrtinvQ @ Bup
        U1, _, V1 = jnp.linalg.svd(Atilde, full_matrices=False)
        U2, _, V2 = jnp.linalg.svd(Btilde, full_matrices=False)
        # Retract
        retracted_params["left"][key]["weight"] = (U1 @ V1) @ sqrtQ
        retracted_params["right"][key]["weight"] = sqrtQ @ (U2 @ V2)
    return retracted_params


def optimal_loss(target, pretrained, L, lora_rank):
    E = target - pretrained
    U, S, V = jnp.linalg.svd(E)
    return 0.5 * jnp.sum(S[L * lora_rank :] ** 2)


def full_training(
    mode,  # "linear", "lora", "cola", "implicit_cola", "hybrid_balora"
    n_steps,  # total number of steps
    trainable_params,
    frozen_params,
    target,
    key,  # for init of each new LoRA (if mode == "cola")
    shapes,
    lora_rank,
    learning_rate=1e-3,
    select_best_learning_rate=False,  # if True, grid search for each LoRA to find the best LR
    grid=jnp.linspace(1e-5, 1e-2, 10),  # tested learning rates
    store_checkpoints=False,
    freq_checkpoints=1000,
    cola_update_period=None,  # number of steps of each LoRA (if mode == "cola")
    random_init=True,  # if True, key is split at each new LoRA init
    verbose=False,
    store_limiting_points=False,
    right_scaling=1,
    left_scaling=0,
    criterion="final",
    init_type="random",
    project_on_balanced="best",
    threshold=1e5,
    project_grad=False,
    switch_to_balora_after=0,
    left_multiplier=False,
    right_multiplier=False,
    adam=False,
):
    if mode in ["cola", "lora", "hybrid_balora"]:
        apply_fn = apply_lora_network
        get_trainable_params_fn = get_trainable_parameters_lora
        if mode == "cola" or mode == "hybrid_balora":
            init_fn = functools.partial(
                init_lora_network,
                shapes=shapes,
                lora_rank=lora_rank,
                init_mode="cola",
                right_scaling=right_scaling,
                left_scaling=left_scaling,
                init_type=init_type,
                left_multiplier=left_multiplier,
                right_multiplier=right_multiplier,
            )
    if mode == "linear":
        apply_fn = apply_linear
        get_trainable_params_fn = get_trainable_parameters_linear
    if mode == "implicit_cola":
        raise NotImplementedError

    def loss_fn(trainable_params, frozen_params):
        output = apply_fn(trainable_params, frozen_params)
        return 0.5 * jnp.sum((output - target) ** 2)

    loss_and_grad = jax.value_and_grad(loss_fn)

    def train_step(trainable_params, frozen_params, learning_rate, project_on_balanced, project_grad, adam):
        loss, grad = loss_and_grad(trainable_params, frozen_params)
        # params = params - learning_rate * grad
        if project_grad:
            grad = project_gradient(trainable_params, grad)
        if adam:
            eps = 1e-8
            grad = jax.tree_util.tree_map(
                lambda g: g / (jnp.abs(g) + eps),
                grad
)
        updated_trainable_params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g, trainable_params, grad
        )
        if project_on_balanced:
            if project_on_balanced == "retract":
                updated_trainable_params = retraction(updated_trainable_params, trainable_params)
            else:
                updated_trainable_params = project_balanced(updated_trainable_params, mode=project_on_balanced)
        return updated_trainable_params, loss

    train_step = jax.jit(
        train_step,
        static_argnames=["project_on_balanced", "project_grad", "adam"],
    )

    def several_train_steps(
        trainable_params, frozen_params, n_steps, learning_rate, threshold=threshold, store_checkpoints=False, project_on_balanced=project_on_balanced, project_grad=project_grad, adam=adam,
    ):
        loss_list = []
        checkpoints_list = []
        for i in range(n_steps):
            if store_checkpoints and i % freq_checkpoints == 0:
                checkpoints_list.append(trainable_params)
            trainable_params, loss = train_step(
                trainable_params, frozen_params, learning_rate, project_on_balanced, project_grad, adam
            )
            loss_list.append(loss.item())
            if loss > threshold:
                loss_list[-1] = threshold
                f"Warning: step size leads to divergence, putting final loss to threshold {threshold}"
                break
        return trainable_params, loss_list, checkpoints_list

    def best_learning_rate_lora(
        trainable_params, frozen_params, n_steps, grid, criterion="threshold_1e-3", project_on_balanced=project_on_balanced, project_grad=project_grad,
    ):
        pretrained = apply_fn(trainable_params, frozen_params)
        L = len(shapes)

        best_val = jnp.inf
        best_learning_rate = None
        output_list = None
        output_params = None
        for learning_rate in grid:
            new_trainable_params, loss_list, _ = several_train_steps(
                trainable_params,
                frozen_params,
                n_steps,
                learning_rate,
                store_checkpoints=False,
                project_on_balanced=project_on_balanced,
                project_grad=project_grad,
                adam=adam,
            )
            if len(loss_list) < n_steps:
                if verbose:
                    print("learning rate:", learning_rate, "above threshold, skipping")
                continue
            if verbose:
                print("learning_rate:", learning_rate, "final_loss:", loss_list[-1])
            if criterion.startswith("final"):
                if "_" in criterion:
                    idx = int(criterion.split("_")[-1])
                else:
                    idx = -1
                val = loss_list[idx]
            elif criterion == "area":
                val = jnp.sum(jnp.array(loss_list))
            elif criterion.startswith("threshold"):
                threshold = float(criterion.split("_")[-1])
                val = np.sum(
                    np.nan_to_num(np.array(loss_list), nan=np.inf, posinf=np.inf)
                    > optimal_loss(target, pretrained, L, lora_rank) + threshold
                )
            if val < best_val:
                best_val = val
                output_list = loss_list
                output_params = new_trainable_params
                best_learning_rate = learning_rate
        if best_val == len(loss_list):
            print("WARNING: threshold is smaller than min of the loss")

        return best_learning_rate, output_list, output_params
    if mode == "lora":
        n_step_list = [n_steps]
    elif mode == "cola":
        n_step_list = [cola_update_period] * (n_steps // cola_update_period)
        if n_steps % cola_update_period:
            n_step_list.append(n_steps % cola_update_period)
    elif mode == "hybrid_balora":
        n_step_list = [switch_to_balora_after, n_steps - switch_to_balora_after] if switch_to_balora_after > 0 else [n_steps]

    losses = []
    checkpoints = []
    frozen_params_list = [frozen_params]
    limiting_points = []
    learning_rate_list = []  # filled only if select_best_learning_rate == True
    for j, n_steps_slice in enumerate(n_step_list):
        if mode == "hybrid_balora" and (j == 1 or len(n_step_list) == 1):
            project_on_balanced = "old"
            if verbose: print("Switching to BaLoRA training")
            if j == 1:
                trainable_params = final_trainable_params  # from previous LoRA phase
        if select_best_learning_rate:
            learning_rate, loss_list, final_trainable_params = best_learning_rate_lora(
                trainable_params,
                frozen_params,
                n_steps_slice,
                grid,
                criterion=criterion,
                project_on_balanced=project_on_balanced,
                project_grad=project_grad,
            )
            if loss_list is None:
                print("Warning: no step-size leads to convergence, skipping")
                break
            learning_rate_list.append(learning_rate)
            print(f"metastep: {j} best_learning_rate: {learning_rate}")
            if learning_rate in [grid[0], grid[-1]]:
                print("WARNING: best_learning_rate is on the edge of the grid")
                # break
        else:
            final_trainable_params, loss_list, _ = several_train_steps(
                trainable_params,
                frozen_params,
                n_steps_slice,
                learning_rate,
                threshold,
                store_checkpoints=False,
                project_on_balanced=project_on_balanced,
                project_grad=project_grad,
                adam=adam,
            )
            if len(loss_list) < n_steps_slice:
                print("Warning: the step-size leads to divergence")
        losses.extend(loss_list)
        if store_checkpoints:
            _, loss_list, checkpoints_list = several_train_steps(
                trainable_params,
                frozen_params,
                n_steps_slice,
                learning_rate,
                threshold,
                store_checkpoints=store_checkpoints,
                project_on_balanced=project_on_balanced,
                project_grad=project_grad,
                adam=adam,
            )
            checkpoints.extend(checkpoints_list)
        if store_limiting_points:
            limiting_points.append(final_trainable_params)

        if mode == "cola":
            new_frozen_params = merge_params(final_trainable_params, frozen_params)
            frozen_params_list.append(new_frozen_params)
            if random_init:
                key, _ = jax.random.split(key)
            new_params = init_fn(key, frozen_params=new_frozen_params)
            trainable_params, frozen_params = get_trainable_params_fn(new_params)

    return (
        trainable_params,
        frozen_params,
        losses,
        frozen_params_list,
        checkpoints,
        limiting_points,
        learning_rate_list,
    )