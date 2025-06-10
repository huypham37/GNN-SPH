"""External force function."""

from typing import Optional

import jax
import jax.numpy as jnp


def force_fn(r: jax.Array, t: Optional[float] = None, **kwargs) -> jax.Array:
    """Force function for 2D reverse Poiseuille flow.

    Args:
        r: Particle position. Shape: (2,)
        t: Time.
        **kwargs: Keyword arguments.
    """

    return jnp.where(
        r[1] > 1.0,
        jnp.array([-1.0, 0.0]),
        jnp.array([1.0, 0.0]),
    )
