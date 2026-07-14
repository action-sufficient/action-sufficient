from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import os
import imageio


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    env_name=None,
    gif_name=None,
    goal_conditioned=True,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    render_video=False,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        env_name: Environment name.
        goal_conditioned: Whether to do goal-conditioned evaluation.
        task_id: Task ID to be passed to the environment (only used when goal_conditioned is True).
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes):
        traj = defaultdict(list)
        should_render = (render_video and (i == 0))

        if goal_conditioned:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
        else:
            observation, info = env.reset()
            goal = None
            goal_frame = None
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render:
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        # save the frames as a GIF
        if should_render:
            os.makedirs('tmp', exist_ok=True)
            imageio.mimsave(f'tmp/{gif_name}_{task_id}.gif', render, fps=30)
        
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_one_episode(
    agent,
    env,
    output_path,
    goal_conditioned=True,
    task_id=None,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Run exactly one evaluation episode and save the rendered rollout.

    Args:
        agent: Agent.
        env: Environment.
        output_path: Output video path (.gif or .mp4).
        goal_conditioned: Whether to pass task_id on reset.
        task_id: Task ID for goal-conditioned environments.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Optional Gaussian noise std for actions.

    Returns:
        A tuple (stats, traj) with episode statistics and trajectory.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    traj = defaultdict(list)

    if goal_conditioned:
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=True))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
    else:
        observation, info = env.reset()
        goal = None
        goal_frame = None

    done = False
    render = []
    while not done:
        action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
        action = np.array(action)
        if eval_gaussian is not None:
            action = np.random.normal(action, eval_gaussian)
        action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render().copy()
        if goal_frame is not None:
            render.append(np.concatenate([goal_frame, frame], axis=0))
        else:
            render.append(frame)

        transition = dict(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )
        add_to(traj, transition)
        observation = next_observation

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(output_path, render, fps=30)

    stats = flatten(info)
    return stats, traj

import gymnasium
import ogbench.manipspace

def evaluate_low_level(
    agent,
    env_name, # cube-single-play-v0, cube-double-play-v0, ...
    task_id=None,
    config=None,
    num_eval_episodes=10,
    num_video_episodes=0,
    video_frame_skip=3,
    render_video=False,
    eval_temperature=0,
    eval_gaussian=None,
    low_level_subgoal_steps=10, # only used if config['agent_name'] not in ['sharsa', 'hiql']
):
    # manually set env for low-level evaluation
    env_name = env_name.replace('-play', '')
    env_name = env_name.replace('-noisy', '')
    if 'cube' in env_name:
        env = gymnasium.make(
            env_name,
            terminate_at_goal=True,
            mode='task',
            permute_blocks=False,
        )
        num_cubes = env.unwrapped._num_cubes
    else:
        env = gymnasium.make(
            env_name,
            terminate_at_goal=True,
            mode='task',
        )

    # use GCIQL agent for now.
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes):
        traj = defaultdict(list)
        should_render = (render_video and (i == 0))
        
        # fixed initial cube position
        if "cube" in env_name:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render, random_init_cube=False))
        else:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []

        # fetch oracle trajectory, set subgoal sequence
        oracle_traj_path = os.path.join('oracle-trajectories', env_name, f'{task_id}.npz')
        oracle_traj = np.load(oracle_traj_path, allow_pickle=True)
        if config['agent_name'] == 'sharsa':
            subgoal_steps = config['subgoal_steps']
        elif 'hiql' in config['agent_name']:
            subgoal_steps = config['actor_subgoal_steps']
        else:
            subgoal_steps = low_level_subgoal_steps
        subgoal_sequence = oracle_traj['observations'][25::subgoal_steps]
        subgoal_it = iter(subgoal_sequence)
        subgoal = next(subgoal_it)

        while not done:
            # if agent achieves the optimal subgoal, update the subgoal
            # if close(observation, subgoal, num_cubes):
            if step % subgoal_steps == 0 and step != 0:
                try:
                    subgoal = next(subgoal_it)
                except StopIteration:
                    subgoal = goal
            
            # pass subgoal to agent instead of goal
            if 'gciql' in config['agent_name'] or config['agent_name'] == 'gcivl':
                action = actor_fn(observations=observation, goals=subgoal, temperature=eval_temperature)
            elif 'hiql' in config['agent_name'] or 'ota' in config['agent_name'] or config['agent_name'] == 'sharsa':
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature, subgoals=subgoal)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render:
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        # save the frames as a GIF
        if should_render:
            os.makedirs('tmp', exist_ok=True)
            imageio.mimsave(f'tmp/{env_name}_low_actor_{task_id}.gif', render, fps=30)

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
    
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
