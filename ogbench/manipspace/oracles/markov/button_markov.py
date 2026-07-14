import numpy as np

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class ButtonMarkovOracle(MarkovOracle):
    def __init__(self, max_step=100, gripper_always_closed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step
        self._gripper_always_closed = gripper_always_closed
        self.target_button = None
        self.target_button_state = None

    def reset(self, ob, info, target_button=None):
        self._done = False
        self._step = 0
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)
        
        if target_button is not None:
            self.target_button = target_button
            self.target_button_state = (info[f'privileged/button_{target_button}_state'] + 1) % 2

    def select_action(self, ob, info):
        if self.target_button is not None:
            env = self._env.unwrapped
            info['privileged/target_button'] = self.target_button
            button_site_id = env._button_site_ids[self.target_button]
            info['privileged/target_button_top_pos'] = env._data.site_xpos[button_site_id].copy()
            info['privileged/target_button_state'] = self.target_button_state
            self._final_pos = np.append(info['privileged/target_button_top_pos'][:-1], 0.25)
            self._final_yaw = info['proprio/effector_yaw'][0]

        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]

        target_button = info['privileged/target_button']
        button_target_top_pos = info['privileged/target_button_top_pos'] + np.array([0, 0, 0.06])
        button_target_bottom_pos = info['privileged/target_button_top_pos'] - np.array([0, 0, 0.022])
        button_state = info[f'privileged/button_{target_button}_state']
        target_state = info['privileged/target_button_state']

        above_threshold = 0.16
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(button_target_top_pos[:2] - effector_pos[:2]) <= 0.04
        target_achieved = button_state == target_state
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_achieved:
            if not xy_aligned:
                self.print_phase('1: Move above the button')
                action = np.zeros(5)
                diff = button_target_top_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[4] = 1
            else:
                self.print_phase('2: Press the button')
                action = np.zeros(5)
                diff = button_target_bottom_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[4] = 1
        else:
            if not above:
                self.print_phase('3: Release the button')
                diff = (
                    np.array([button_target_top_pos[0], button_target_top_pos[1], above_threshold * 2]) - effector_pos
                )
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = 1 if self._gripper_always_closed else -1
            else:
                self.print_phase('4: Move to the final position')
                diff = self._final_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = 1 if self._gripper_always_closed else -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if self._debug:
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action
