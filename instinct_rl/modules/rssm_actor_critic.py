from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch

from .rssm import RSSMWorldModel
from instinct_rl.utils.utils import get_subobs_size


class RSSMActorCriticMixin:
    """Append RSSM deterministic memory to actor and critic observations."""

    def __init__(
        self,
        obs_format: Dict[str, OrderedDict[str, tuple]],
        num_actions,
        rssm_config: dict,
        rssm_feature_name: str = "rssm_feature",
        **kwargs,
    ):
        self.raw_obs_segments = obs_format["policy"]
        self.raw_critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        self.raw_obs_dim = get_subobs_size(self.raw_obs_segments)
        self.rssm_feature_name = rssm_feature_name
        rssm_config = rssm_config.copy()
        rssm_feature_dim = rssm_config.get("deter_dim", 256)

        rssm_obs_format = deepcopy(obs_format)
        rssm_obs_format["policy"] = OrderedDict(rssm_obs_format["policy"])
        rssm_obs_format["policy"][rssm_feature_name] = (rssm_feature_dim,)
        if "critic" in rssm_obs_format:
            rssm_obs_format["critic"] = OrderedDict(rssm_obs_format["critic"])
            rssm_obs_format["critic"][rssm_feature_name] = (rssm_feature_dim,)

        super().__init__(rssm_obs_format, num_actions, **kwargs)
        self.rssm = RSSMWorldModel(self.raw_obs_segments, num_actions, **rssm_config)
        self._rssm_state = None
        self._rssm_prev_action = None
        self._last_rssm_feature = None

    def _append_rssm_feature(self, observations: torch.Tensor, rssm_features: torch.Tensor | None):
        if rssm_features is None:
            if observations.shape[-1] == self.raw_obs_dim:
                feature, self._rssm_state = self.rssm.step_feature(
                    observations,
                    self._get_prev_action(observations),
                    self._rssm_state,
                )
            else:
                feature = torch.zeros(observations.shape[0], self.rssm.feature_dim, device=observations.device)
        else:
            feature = rssm_features
        self._last_rssm_feature = feature.detach()
        return torch.cat([observations, feature], dim=-1)

    def _get_prev_action(self, observations: torch.Tensor):
        if self._rssm_prev_action is None or self._rssm_prev_action.shape[0] != observations.shape[0]:
            self._rssm_prev_action = torch.zeros(observations.shape[0], self.rssm.num_actions, device=observations.device)
        return self._rssm_prev_action

    def act(self, observations, rssm_features=None, **kwargs):
        observations = self._append_rssm_feature(observations, rssm_features)
        actions = super().act(observations, **kwargs)
        if rssm_features is None:
            self._rssm_prev_action = actions.detach()
        return actions

    def act_inference(self, observations, rssm_features=None):
        observations = self._append_rssm_feature(observations, rssm_features)
        actions = super().act_inference(observations)
        if rssm_features is None:
            self._rssm_prev_action = actions.detach()
        return actions

    def evaluate(self, critic_observations, rssm_features=None, **kwargs):
        critic_observations = self._append_rssm_feature(critic_observations, rssm_features)
        return super().evaluate(critic_observations, **kwargs)

    def reset(self, dones=None):
        self._rssm_state = self.rssm.reset_state(self._rssm_state, dones)
        if self._rssm_prev_action is not None and dones is not None:
            self._rssm_prev_action = self._rssm_prev_action * (1.0 - dones.float()).view(-1, 1)
        return super().reset(dones)

    def get_and_clear_last_rssm_feature(self):
        return self._last_rssm_feature
