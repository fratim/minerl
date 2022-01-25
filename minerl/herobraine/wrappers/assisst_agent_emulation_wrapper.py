__author__ = "Tim Franzmeyer"
__email__ = "tfranzmeyer [at] gmail [dot] com"

import threading
import gym
import numpy as np


class AssistWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.assist_agent = "agent_0"
        self.lead_agent = "agent_1" # this agent is emulated within this wrapper

        self.action_space = self.env.action_space[self.assist_agent]

    def get_lead_agent_action(self):
        return self.env.action_space[self.lead_agent].sample()

    def step(self, action_in_assist_agent):
        assert self.env.task.agent_count <= 2
        if self.env.task.agent_count == 2: # allows to also use this for single agent environments
            lead_agent_action = self.get_lead_agent_action()
            action = {
                self.assist_agent: action_in_assist_agent,
                self.lead_agent: lead_agent_action
            }
        # else:
        #     action = {
        #         self.assist_agent: action_in_assist_agent
        #     }
        else:
            action = action_in_assist_agent

        obs, rew, done, info = self.env.step(action)

        if self.env.task.agent_count == 2:
            obs = obs[self.assist_agent]
            rew = rew[self.assist_agent]
            info = info[self.assist_agent]

        processed_reward = self.post_process_reward(rew, obs)

        return obs, processed_reward, done, info

    def post_process_reward(self, reward, obs):

        assert reward == 0
        target_pos = np.array([0, 0, 0])
        if "raw" in obs.keys():
            cur_location = obs["raw"]["location_stats"]
        else:
            cur_location = obs["location_stats"]

        cur_pos = np.array([cur_location["xpos"], cur_location["ypos"], cur_location["zpos"]])

        assert np.abs(cur_location["xpos"]) < 10000 and np.abs(cur_location["ypos"]) < 10000 and np.abs(cur_location["zpos"])< 10000

        delta = np.linalg.norm(target_pos - cur_pos)

        reward = -1*delta

        return reward




    def reset(self):
        multi_obs = self.env.reset()
        if self.env.task.agent_count > 1:
            return multi_obs[self.assist_agent]
        else:
            return multi_obs