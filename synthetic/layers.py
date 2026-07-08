import functools

import jax
import jax.numpy as jnp


def init_linear(key, shape):
    return {"weight": jax.random.normal(key, shape)}


def init_linear_network(key, shapes, mode="random"):
    ### shapes is a list of compatible shapes, of length depth.
    # mode in ["random", "eye", "zero"]
    params = {}
    for i, shape in enumerate(shapes):
        if mode == "eye":
            params[f"layer_{i}"] = {"weight": jnp.eye(shape[0])}
        elif mode == "random":
            key, subkey = jax.random.split(key)
            params[f"layer_{i}"] = init_linear(key, shape)
        elif mode == "zero":
            params[f"layer_{i}"] = {"weight": jnp.zeros(shape)}
    return params


def apply_linear(params, frozen_params=None):
    del frozen_params
    depth = len(params)
    matrix = params["layer_0"]["weight"]
    for i in range(1, depth):
        matrix = params[f"layer_{i}"]["weight"] @ matrix
    return matrix


def init_lora(
    key, shape, lora_rank, right_scaling=1, left_scaling=0, init_type="random",
):
    p, q = shape
    if init_type == "random":
        key, subkey = jax.random.split(key)
        return {
            "left_weight": jax.random.normal(key, (p, lora_rank)) * left_scaling,
            "right_weight": jax.random.normal(subkey, (lora_rank, q)) * right_scaling,
        }
    if init_type == "unbalanced":
        return {
            "left_weight": jnp.zeros((p, lora_rank)),
            "right_weight": jnp.eye(lora_rank, q) * right_scaling,
        }
    if init_type == "balanced":
        assert left_scaling == right_scaling
        return {
            "left_weight": jnp.eye(p, lora_rank) * left_scaling,
            "right_weight": jnp.eye(lora_rank, q) * right_scaling,
        }
    if init_type == "orthogonal":
        assert p == q, "Orthogonal initialization requires a square target"
        assert 2 * lora_rank <= p, "Lora rank must be less than p / 2 for orthogonal initialization"
        key, subkey = jax.random.split(key)
        random_matrix = jax.random.normal(key, (p, 2 * lora_rank))
        Q, _ = jnp.linalg.qr(random_matrix)
        return {
            "left_weight": Q[:, :lora_rank] * right_scaling ** 0.5,
            "right_weight": Q[:, lora_rank:2 * lora_rank].T * right_scaling ** 0.5,
        }
    if init_type == "zero":
        return {
            "left_weight": jnp.zeros((p, lora_rank)),
            "right_weight": jnp.zeros((lora_rank, q)),
        }


def init_lora_network(
    key,
    shapes,
    lora_rank,
    init_mode,
    frozen_params=None,
    mode="random",
    right_scaling=1,
    left_scaling=0,
    init_type="random",
    left_multiplier=False,
    right_multiplier=False,
    target=None,
):
    params = {}
    lora_params = {
            "left": {},
            "right": {},
        }
    if init_mode == "loraga" or init_mode == "frank_wolfe" or init_mode == "zero_loraga":

        def loss_fn(frozen_params):
            output = apply_linear(frozen_params)
            return 0.5 * jnp.sum((output - target) ** 2)
        
        loss_and_grad = jax.value_and_grad(loss_fn)
        loss, grad = loss_and_grad(frozen_params)
        for l in range(len(shapes)):
            if (l == 0 and right_multiplier) or (l == len(shapes) - 1 and left_multiplier):
                lora_params["left"][f"layer_{l}"] = {"weight": jnp.zeros((shapes[l][0], lora_rank))}
                lora_params["right"][f"layer_{l}"] = {"weight": jnp.zeros((lora_rank, shapes[l][1]))}
            else:
                g_l = grad[f"layer_{l}"]["weight"]
                U, s, V = jnp.linalg.svd(g_l, full_matrices=False)
                lora_params["left"][f"layer_{l}"] = {
                    "weight": left_scaling * jnp.array(U[:, :lora_rank]) @ jnp.diag(jnp.array(s[:lora_rank]) ** 0.5)
                }
                lora_params["right"][f"layer_{l}"] = {
                    "weight": right_scaling * jnp.diag(jnp.array(s[:lora_rank]) ** 0.5) @ jnp.array(V[:lora_rank, :])
                }
        
        if init_mode == "loraga" or init_mode == "zero_loraga":
            
            def substract_lora(frozen_array, left_lora, right_lora):
                return frozen_array - left_lora @ right_lora
            
            frozen_params = jax.tree_util.tree_map(
                substract_lora, frozen_params, lora_params["left"], lora_params["right"]
            )
        params["frozen"] = frozen_params
    if not (init_mode == "loraga" or init_mode == "frank_wolfe"):
        for i, shape in enumerate(shapes):
            key, subkey = jax.random.split(key)
            if (i == 0 and right_multiplier) or (i == len(shapes) - 1 and left_multiplier):
                lora_weights = init_lora(
                    key,
                    shape,
                    lora_rank,
                    right_scaling=right_scaling,
                    left_scaling=left_scaling,
                    init_type="zero",
                )
            else:
                lora_weights = init_lora(
                    key,
                    shape,
                    lora_rank,
                    right_scaling=right_scaling,
                    left_scaling=left_scaling,
                    init_type=init_type,
                )
            for side in ["left", "right"]:
                lora_params[side][f"layer_{i}"] = {"weight": lora_weights[f"{side}_weight"]}
    params["lora_params"] = lora_params
    if init_mode == "lora":
        key, subkey = jax.random.split(key)
        params["frozen"] = init_linear_network(key, shapes, mode=mode)
    elif init_mode == "cola":
        params["frozen"] = frozen_params
    return params


def merge_params(trainable_params, frozen_params):
    def merge_lora(frozen_array, left_lora, right_lora):
        return frozen_array + left_lora @ right_lora

    merged_tree = jax.tree_util.tree_map(
        merge_lora, frozen_params, trainable_params["left"], trainable_params["right"]
    )
    return merged_tree


def apply_lora_network(lora_params, frozen_params):
    merged_tree = merge_params(lora_params, frozen_params)
    return apply_linear(merged_tree)


def get_trainable_parameters_lora(params):
    return params["lora_params"], params["frozen"]


def get_trainable_parameters_linear(params):
    return params, None


def get_finetuning_fns(shapes, lora_rank):
    init_fn = functools.partial(init_lora_network, shapes=shapes, lora_rank=lora_rank)
    return init_fn, apply_lora_network, get_trainable_parameters_lora


def get_pretraining_fns(shapes):
    init_fn = functools.partial(init_linear_network, shapes=shapes)
    return init_fn, apply_linear, get_trainable_parameters_linear


if __name__ == "__main__":
    shapes = (
        (3, 2),
        (4, 3),
        (3, 4),
    )
    lora_rank = 2
    target_shape = (shapes[-1][0], shapes[0][1])
    key = jax.random.PRNGKey(0)
    params = init_lora_network(key, shapes, lora_rank)
    apply_lora_network(params["lora_params"], params["frozen"])