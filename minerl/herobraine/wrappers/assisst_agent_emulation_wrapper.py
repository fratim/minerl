__author__ = "Tim Franzmeyer"
__email__ = "tfranzmeyer [at] gmail [dot] com"

import threading
import gym


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

        lead_agent_action = self.get_lead_agent_action()

        multi_action = {
            self.lead_agent: lead_agent_action,
            self.assist_agent: action_in_assist_agent
        }

        obs, rew, done, info = self.env.step(multi_action)
        return obs[self.assist_agent], rew[self.assist_agent], done, info[self.assist_agent]

    def reset(self):
        multi_obs = self.env.reset()
        return multi_obs[self.assist_agent]