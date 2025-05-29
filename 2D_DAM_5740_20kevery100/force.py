"""External force function."""

from typing import Optional

import jax
import jax.numpy as jnp


def force_fn(r: jax.Array, t: Optional[float] = None, **kwargs) -> jax.Array:
    """Force function for 2D dam break.

    Args:
        r: Particle position. Shape: (2,)
        t: Time.
        **kwargs: Keyword arguments.
    """

    return jnp.array([0.0, -1.0])
