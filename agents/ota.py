from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCValue, Identity, LengthNormalize, OracleRepLayer


class OTAAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning (HIQL) agent.

    Changes:
        - added n-step TD. (already added.)
        - added which goal rep to use.
        - added option whether to use general goals or subgoals for low-level actor.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def low_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_low_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_low_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('low_value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_high_value')(
            batch['high_value_option_observations'], batch['high_value_goals']
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['high_value_rewards'] + self.config['discount'] * batch['high_value_masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_high_value')(batch['observations'], batch['high_value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['high_value_rewards'] + self.config['discount'] * batch['high_value_masks'] * next_v1_t
        q2 = batch['high_value_rewards'] + self.config['discount'] * batch['high_value_masks'] * next_v2_t
        (v1, v2) = self.network.select('high_value')(batch['observations'], batch['high_value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        actor_goals = batch['high_actor_goals'] if self.config['general_goal'] else batch['low_actor_goals']

        v1, v2 = self.network.select('low_value')(batch['observations'], actor_goals)
        nv1, nv2 = self.network.select('low_value')(batch['next_observations'], actor_goals)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        if self.config['use_value_rep'] or self.config['use_actor_rep'] or self.config['use_oracle_rep']:
            # Compute the goal representations of the subgoals.
            goal_reps = self.network.select('goal_rep')(
                jnp.concatenate([batch['observations'], actor_goals], axis=-1),
                params=grad_params,
            )
            if not self.config['use_actor_rep']:
                # Stop gradients through the goal representations.
                goal_reps = jax.lax.stop_gradient(goal_reps)
        else:
            goal_reps = actor_goals

        is_goal_encoded = self.config['use_value_rep'] or self.config['use_actor_rep'] or self.config['use_oracle_rep']

        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=is_goal_encoded, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        v1, v2 = self.network.select('high_value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('high_value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        if self.config['use_value_rep'] or self.config['use_actor_rep']:
            target = self.network.select('goal_rep')(
                jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
            )
        else:
            target = batch['high_actor_targets']
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        if self.config['abstraction_factor'] == 1:
            high_value_loss = 0.0
            high_value_info = {}
        else:
            high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = low_value_loss + high_value_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'low_value')

        if self.config['abstraction_factor'] == 1:
            new_network.params['modules_high_value'] = new_network.params['modules_low_value']
            new_network.params['modules_target_high_value'] = new_network.params['modules_target_low_value']
        else:
            self.target_update(new_network, 'high_value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        subgoals=None,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        if subgoals is None:
            high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
            goal_reps = high_dist.sample(seed=high_seed)
            if self.config['use_value_rep'] or self.config['use_actor_rep']:
                goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        else:
            if self.config['use_value_rep'] or self.config['use_actor_rep']:
                goal_reps = self.network.select('goal_rep')(
                    jnp.concatenate([observations, subgoals], axis=-1)
                )
            else:
                goal_reps = subgoals

        is_goal_encoded = self.config['use_value_rep'] or self.config['use_actor_rep'] or self.config['use_oracle_rep']
        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=is_goal_encoded, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        actions = jnp.clip(actions, -1, 1)

        return actions

    @jax.jit
    def get_value(self, observations, goals, actions):
        v1, v2 = self.network.select('low_value')(observations, goals)
        v = (v1 + v2) / 2
        return v, None

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
        env_name=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        # assert config['use_value_rep'] or config['use_actor_rep'], "At least one of them must be True."

        if config['use_oracle_rep']:
            goal_rep_def = OracleRepLayer(env_name=env_name)
            config['rep_dim'] = goal_rep_def(jnp.concatenate([ex_observations, ex_goals], axis=-1)).shape[-1]

            low_value_encoder_def = None
            target_low_value_encoder_def = None
            high_value_encoder_def = None
            target_high_value_encoder_def = None
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None
        elif config['use_value_rep']:
            # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
            if config['encoder'] is not None:
                encoder_module = encoder_modules[config['encoder']]
                goal_rep_seq = [encoder_module()]
            else:
                goal_rep_seq = []
            goal_rep_seq.append(
                MLP(
                    hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                )
            )
            goal_rep_seq.append(LengthNormalize())
            goal_rep_def = nn.Sequential(goal_rep_seq)

            # Define the encoders that handle the inputs to the value and actor networks.
            # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
            # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
            # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
            # Value: V(s, phi([s; g]))
            if config['encoder'] is not None:
                low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                target_low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                high_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                target_high_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                # Low-level actor: pi^l(. | s, phi([s; w]))
                low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                # High-level actor: pi^h(. | s, g) (i.e., no encoder)
                high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
            else:
                low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                target_low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                high_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                target_high_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                # Low-level actor: pi^l(. | s, phi([s; w]))
                low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                # High-level actor: pi^h(. | s, g) (i.e., no encoder)
                high_actor_encoder_def = None
        elif config['use_actor_rep']:
            if config['encoder'] is not None:
                encoder_module = encoder_modules[config['encoder']]
                goal_rep_seq = [encoder_module()]
            else:
                goal_rep_seq = []
            goal_rep_seq.append(
                MLP(
                    hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                )
            )
            goal_rep_seq.append(LengthNormalize())
            goal_rep_def = nn.Sequential(goal_rep_seq)

            if config['encoder'] is not None:
                low_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
                target_low_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
                high_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
                target_high_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
                low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
                high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
            else:
                low_value_encoder_def = None
                target_low_value_encoder_def = None
                high_value_encoder_def = None
                target_high_value_encoder_def = None
                low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
                high_actor_encoder_def = None
        else:
            low_value_encoder_def = None
            target_low_value_encoder_def = None
            high_value_encoder_def = None
            target_high_value_encoder_def = None
            low_actor_encoder_def = None
            high_actor_encoder_def = None

        # Define networks.
        low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=low_value_encoder_def,
        )
        target_low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=target_low_value_encoder_def,
        )

        high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=high_value_encoder_def,
        )
        target_high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=target_high_value_encoder_def,
        )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'] if (config['use_value_rep'] or config['use_actor_rep']) else goal_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=low_actor_encoder_def,
        )

        network_info = dict(
            low_value=(low_value_def, (ex_observations, ex_goals)),
            target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            target_high_value=(target_high_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        if config['use_value_rep'] or config['use_actor_rep'] or config['use_oracle_rep']:
            network_info.update(
                goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_low_value'] = params['modules_low_value']

        if config['abstraction_factor'] == 1:
            params['modules_high_value'] = params['modules_low_value']
            params['modules_target_high_value'] = params['modules_target_low_value']
        else:
            params['modules_target_high_value'] = params['modules_high_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='ota',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            use_oracle_rep=False,  # Whether to use oracle goal representation (True) or learned representation (False).
            use_value_rep=True,  # Whether to use value goal representation.
            use_actor_rep=False,  # Whether to use actor goal representation.
            rep_dim=10,  # Goal representation dimension.
            # low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            general_goal=False,  # Whether low-level actor uses general goals (True) or subgoals (False).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset_ota',  # Dataset class name.
            abstraction_factor=1,  # Value subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            subgoal_steps=25,  # Actor subgoal steps.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
