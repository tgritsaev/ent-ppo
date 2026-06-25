from math import prod
from itertools import product as itertools_product

import chex
import jax
import jax.numpy as jnp

from ..base import TRewardModule
from ..utils import QM9_SMALL_BLOCKS, QM9_SMALL_FULL_ALPHABET
from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    FixedPrependAppendSequenceEnvironment,
)


class QM9SmallEnvironment(FixedPrependAppendSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {char: i for i, char in enumerate(QM9_SMALL_FULL_ALPHABET)}

        super().__init__(
            reward_module,
            max_length=5,
            nchar=len(QM9_SMALL_BLOCKS),
            ntoken=len(QM9_SMALL_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )

        offsets = [0]
        for k in range(1, self.max_length + 1):
            offsets.append(offsets[-1] + self.nchar ** (k - 1))
        self._length_offsets = offsets

        self._num_states = offsets[-1] + self.nchar ** self.max_length

    @property
    def name(self) -> str:
        """Environment name."""
        return "QM9Small-v0"

    @property
    def is_enumerable(self) -> bool:
        """Whether the environment is enumerable."""
        return True
    
    def state_to_index(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return a unique integer index in [0, num_states) for a single state.

        Indexing scheme:
          - length = number of non-PAD tokens  (0..8)
          - offset = _length_offsets[length]
          - inner  = lexicographic index within the length-block
          - index  = offset + inner
        """
        tokens = state.tokens

        num_pad = jnp.sum(tokens == self.pad_token)
        length = self.max_length - num_pad

        offsets = jnp.array(self._length_offsets, dtype=jnp.int32)
        offset = offsets[length]

        positions = jnp.arange(self.max_length)
        valid = positions < length

        exponents = jnp.where(valid, length - 1 - positions, 0)
        weights = jnp.pow(self.nchar, exponents).astype(jnp.int32)
        weights = jnp.where(valid, weights, 0)

        inner = jnp.sum(tokens * weights)
        return offset + inner
    
    def get_all_states(self, env_params: EnvParams) -> EnvState:
        """Return all 87381 states in state_to_index order.

        Order:
          1. Empty sequence (length 0).
          2. All sequences of lengths 1..8 in lexicographic token order.

        States of length 8 are marked as terminal.

        Returns:
            EnvState with tokens of shape [87381, 8] and flag arrays of shape [87381].
        """
        all_tokens = []
        all_is_terminal = []
        all_is_initial = []

        all_tokens.append([self.pad_token] * self.max_length)
        all_is_terminal.append(False)
        all_is_initial.append(True)
        
        for length in range(1, self.max_length + 1):
            is_terminal = (length == self.max_length)
            for combo in itertools_product(range(self.nchar), repeat=length):
                tokens = list(combo) + [self.pad_token] * (self.max_length - length)
                all_tokens.append(tokens)
                all_is_terminal.append(is_terminal)
                all_is_initial.append(False)

        return EnvState(
            tokens=jnp.array(all_tokens, dtype=jnp.int32),
            is_terminal=jnp.array(all_is_terminal, dtype=jnp.bool_),
            is_initial=jnp.array(all_is_initial, dtype=jnp.bool_),
            is_pad=jnp.zeros(len(all_tokens), dtype=jnp.bool_),
        )

    def _get_states_rewards(self, env_params: EnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states in the hypergrid.
        """
        rewards = jnp.zeros((self.nchar,) * self.max_length, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)  # Unpack index to state
            env_state = EnvState(
                tokens=jnp.array(state),
                is_terminal=True,
                is_initial=False,
                is_pad=False,
            )
            batched_env_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), env_state)
            reward = self.reward_module.reward(batched_env_state, env_params)
            return rewards.at[state].set(reward[0])

        rewards = jax.lax.fori_loop(0, self.nchar**self.max_length, update_rewards, rewards)
        return rewards

    def get_true_distribution(self, env_params: EnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states in the hypergrid.
        """
        rewards = self._get_states_rewards(env_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """
        Extracts the empirical distribution from the given states.
        """
        dist_shape = (self.nchar,) * self.max_length
        sample_idx = jax.vmap(lambda x: jnp.ravel_multi_index(x, dims=dist_shape, mode="clip"))(
            states.tokens
        )

        valid_mask = states.is_terminal.astype(jnp.float32)
        empirical_dist = jax.ops.segment_sum(valid_mask, sample_idx, num_segments=prod(dist_shape))
        empirical_dist = empirical_dist.reshape(dist_shape)
        empirical_dist /= empirical_dist.sum()
        return empirical_dist

    @property
    def is_mean_reward_tractable(self) -> bool:
        """Whether this environment supports mean reward tractability."""
        return True

    def get_mean_reward(self, env_params: EnvParams) -> float:
        """
        Returns the mean reward for the hypergrid environment.
        The mean reward is computed as the sum of rewards divided by the number of states.
        """
        rewards = self._get_states_rewards(env_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return True

    def get_normalizing_constant(self, env_params: EnvParams) -> float:
        """
        Returns the normalizing constant for the hypergrid environment.
        The normalizing constant is computed as the sum of rewards.
        """
        rewards = self._get_states_rewards(env_params)
        return rewards.sum()

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable sampling from the GT distribution."""
        return True

    def get_ground_truth_sampling(
        self, rng_key: chex.PRNGKey, batch_size: int, env_params: EnvParams
    ) -> EnvState:
        """
        Returns a batch of terminal states sampled from the ground-truth distribution
        proportional to rewards over all sequences of length `max_length`.

        Args:
            rng_key: JAX random key for sampling.
            batch_size: Number of samples to generate.
            env_params: Environment parameters.

        Returns:
            EnvState with shape [batch_size, max_length].
        """
        true_distribution = self.get_true_distribution(env_params)
        flat_distribution = true_distribution.flatten()

        sampled_indices = jax.random.choice(
            rng_key,
            a=flat_distribution.size,
            shape=(batch_size,),
            p=flat_distribution,
        )

        sampled_coords_unstacked = jnp.unravel_index(
            sampled_indices, shape=true_distribution.shape
        )
        sampled_tokens = jnp.stack(sampled_coords_unstacked, axis=1)

        return EnvState(
            tokens=sampled_tokens.astype(jnp.int32),
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool_),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool_),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool_),
        )