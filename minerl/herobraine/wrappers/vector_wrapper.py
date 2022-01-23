# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import numpy as np
from functools import reduce
from collections import OrderedDict

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.util import union_spaces, flatten_spaces, intersect_space
from minerl.herobraine.wrapper import EnvWrapper


class Vectorized(EnvWrapper):
    """
    Normalizes and flattens a typical env space for obfuscation.
    common_envs : specified
    """

    def _update_name(self, name: str) -> str:
        return name.split('-')[0] + 'Vector-' + name.split('-')[-1]

    def __init__(self, env_to_wrap: EnvSpec, common_envs=None):
        self.env_to_wrap = env_to_wrap
        self.common_envs = [env_to_wrap] if common_envs is None or len(common_envs) == 0 else common_envs

        # Gather all of the handlers for the common_env
        self.common_actions = reduce(union_spaces, [env.actionables for env in self.common_envs])
        self.flat_actions, self.remaining_action_space = flatten_spaces(self.common_actions)
        self.action_vector_len = sum(space.shape[0] for space in self.flat_actions)
        self.common_action_space = spaces.Dict(
            {agent: spaces.Dict({hdl.to_string(): hdl.space for hdl in self.common_actions}) for agent in self.env_to_wrap.agent_names})

        self.common_observations = reduce(union_spaces, [env.observables for env in self.common_envs])
        self.flat_observations, self.remaining_observation_space = flatten_spaces(self.common_observations)
        self.observation_vector_len = sum(space.shape[0] for space in self.flat_observations)
        self.common_observation_space = spaces.Dict(
            {agent: spaces.Dict({hdl.to_string(): hdl.space for hdl in self.common_observations}) for agent in self.env_to_wrap.agent_names})

        super().__init__(env_to_wrap)

    def _wrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        flat_obs_part = self.common_observation_space[agent].flat_map(obs)
        wrapped_obs = self.common_observation_space[agent].unflattenable_map(obs)
        wrapped_obs['vector'] = flat_obs_part

        return wrapped_obs

    def _wrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        flat_act_part = self.common_action_space[agent].flat_map(act)
        wrapped_act = self.common_action_space[agent].unflattenable_map(act)
        wrapped_act['vector'] = flat_act_part
        return wrapped_act

    def _unwrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        full_obs = self.common_observation_space[agent].unmap_mixed(obs['vector'], obs)
        return intersect_space(self.env_to_wrap.observation_space[agent], full_obs)

    def _unwrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        full_act = self.common_action_space[agent].unmap_mixed(act['vector'], act)
        return intersect_space(self.env_to_wrap.action_space[agent], full_act)


    def create_observation_space(self):

        def get_observation_space_agent():
            obs_list = self.remaining_observation_space
            obs_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.observation_vector_len], dtype=np.float32)))
            return sorted(obs_list)

        ospace = spaces.Dict({aname: spaces.Dict(get_observation_space_agent()) for aname in self.agent_names})
        return ospace

        # for aname in self.agent_names:
        #     obs_list = self.remaining_observation_space
        #     # Todo: add maximum.
        #     obs_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.observation_vector_len], dtype=np.float32)))
        #     ospace[aname] = spaces.Dict(sorted(obs_list))


    def create_action_space(self):
        def get_action_space_agent():
            act_list = self.remaining_action_space
            act_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.action_vector_len], dtype=np.float32)))
            return sorted(act_list)

        aspace = spaces.Dict({aname: spaces.Dict(get_action_space_agent()) for aname in self.agent_names})
        return aspace


    def get_docstring(self):
        return self.env_to_wrap.get_docstring()
