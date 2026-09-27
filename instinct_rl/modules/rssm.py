from collections import OrderedDict
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


def _activation(name: str):
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "lrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported RSSM activation: {name}")


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation: str = "elu"):
    layers = []
    prev_dim = input_dim
    act = _activation(activation)
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(prev_dim, hidden_dim), deepcopy(act)]
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class RSSMWorldModel(nn.Module):
    """A compact RSSM for proprioception + depth prediction.

    The policy consumes only the deterministic recurrent feature. The stochastic
    state and decoders are trained as auxiliary world-model heads.
    """

    def __init__(
        self,
        obs_segments: OrderedDict[str, tuple],
        num_actions: int,
        proprio_components: list[str],
        depth_component: str = "depth_image",
        deter_dim: int = 256,
        stoch_dim: int = 32,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        activation: str = "elu",
        min_std: float = 0.1,
        kl_free: float = 1.0,
        kl_scale: float = 0.1,
        proprio_loss_scale: float = 1.0,
        depth_loss_scale: float = 1.0,
        reward_loss_scale: float = 0.1,
    ):
        super().__init__()
        self.obs_segments = obs_segments
        self.num_actions = num_actions
        self.proprio_components = proprio_components
        self.depth_component = depth_component
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        self.kl_free = kl_free
        self.kl_scale = kl_scale
        self.proprio_loss_scale = proprio_loss_scale
        self.depth_loss_scale = depth_loss_scale
        self.reward_loss_scale = reward_loss_scale

        self.proprio_dim = get_subobs_size(obs_segments, proprio_components)
        self.depth_shape = obs_segments[depth_component]
        self.depth_dim = int(torch.tensor(self.depth_shape).prod().item())

        self.proprio_encoder = _mlp(self.proprio_dim, [hidden_dim], embed_dim // 2, activation)
        self.depth_encoder = nn.Sequential(
            nn.Flatten(),
            _mlp(self.depth_dim, [hidden_dim], embed_dim // 2, activation),
        )
        self.pre_gru = _mlp(stoch_dim + num_actions, [hidden_dim], hidden_dim, activation)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior = _mlp(deter_dim, [hidden_dim], 2 * stoch_dim, activation)
        self.posterior = _mlp(deter_dim + embed_dim, [hidden_dim], 2 * stoch_dim, activation)
        feat_dim = deter_dim + stoch_dim
        self.proprio_decoder = _mlp(feat_dim, [hidden_dim], self.proprio_dim, activation)
        self.depth_decoder = _mlp(feat_dim, [hidden_dim], self.depth_dim, activation)
        self.reward_head = _mlp(feat_dim, [hidden_dim], 1, activation)

    @property
    def feature_dim(self):
        return self.deter_dim

    def initial(self, batch_size: int, device: torch.device):
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def extract_proprio(self, obs: torch.Tensor):
        return get_subobs_by_components(obs, self.proprio_components, self.obs_segments)

    def extract_depth(self, obs: torch.Tensor):
        return get_subobs_by_components(obs, [self.depth_component], self.obs_segments)

    def encode(self, obs: torch.Tensor):
        proprio = self.extract_proprio(obs)
        depth = self.extract_depth(obs)
        return torch.cat([self.proprio_encoder(proprio), self.depth_encoder(depth)], dim=-1)

    def _dist(self, stats: torch.Tensor):
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return Independent(Normal(mean, std), 1)

    def _stats(self, stats: torch.Tensor):
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def obs_step(self, prev_state: dict[str, torch.Tensor], prev_action: torch.Tensor, embed: torch.Tensor):
        x = self.pre_gru(torch.cat([prev_state["stoch"], prev_action], dim=-1))
        deter = self.gru(x, prev_state["deter"])
        prior_mean, prior_std = self._stats(self.prior(deter))
        post_mean, post_std = self._stats(self.posterior(torch.cat([deter, embed], dim=-1)))
        prior_dist = Independent(Normal(prior_mean, prior_std), 1)
        post_dist = Independent(Normal(post_mean, post_std), 1)
        stoch = post_dist.rsample()
        prior = {
            "deter": deter,
            "stoch": prior_dist.rsample(),
            "mean": prior_mean,
            "std": prior_std,
        }
        post = {
            "deter": deter,
            "stoch": stoch,
            "mean": post_mean,
            "std": post_std,
        }
        return post, prior

    def rollout_features(self, obs: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor | None = None):
        """Run RSSM over rollout tensors shaped [T, B, ...]."""
        time, batch = obs.shape[:2]
        state = self.initial(batch, obs.device)
        prev_action = torch.zeros(batch, self.num_actions, device=obs.device)
        posts, priors = [], []
        for t in range(time):
            if dones is not None and t > 0:
                not_done = 1.0 - dones[t - 1].float().view(batch, 1)
                state = {k: v * not_done for k, v in state.items()}
                prev_action = prev_action * not_done
            embed = self.encode(obs[t])
            post, prior = self.obs_step(state, prev_action, embed)
            posts.append(post)
            priors.append(prior)
            state = {"deter": post["deter"], "stoch": post["stoch"]}
            prev_action = actions[t]
        return self._stack_states(posts), self._stack_states(priors)

    def step_feature(self, obs: torch.Tensor, prev_action: torch.Tensor, state: dict[str, torch.Tensor] | None):
        if state is None:
            state = self.initial(obs.shape[0], obs.device)
        embed = self.encode(obs)
        post, _ = self.obs_step(state, prev_action, embed)
        next_state = {"deter": post["deter"].detach(), "stoch": post["stoch"].detach()}
        return post["deter"], next_state

    def reset_state(self, state: dict[str, torch.Tensor] | None, dones: torch.Tensor | None):
        if state is None or dones is None:
            return state
        not_done = (1.0 - dones.float()).view(-1, 1)
        return {k: v * not_done for k, v in state.items()}

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        posts, priors = self.rollout_features(obs, actions, dones)
        feat = torch.cat([posts["deter"], posts["stoch"]], dim=-1)
        proprio_target = self.extract_proprio(obs)
        depth_target = self.extract_depth(obs)
        proprio_pred = self.proprio_decoder(feat)
        depth_pred = self.depth_decoder(feat).reshape_as(depth_target)
        reward_pred = self.reward_head(feat).squeeze(-1)
        reward_target = rewards.mean(dim=-1) if rewards.ndim == 3 else rewards

        proprio_loss = F.mse_loss(proprio_pred, proprio_target)
        depth_loss = F.mse_loss(depth_pred, depth_target)
        reward_loss = F.mse_loss(reward_pred, reward_target)
        kl = torch.distributions.kl_divergence(posts["dist"], priors["dist"])
        kl_loss = torch.clamp(kl, min=self.kl_free).mean()
        loss = (
            self.proprio_loss_scale * proprio_loss
            + self.depth_loss_scale * depth_loss
            + self.reward_loss_scale * reward_loss
            + self.kl_scale * kl_loss
        )
        stats = {
            "rssm_proprio_loss": proprio_loss.detach(),
            "rssm_depth_loss": depth_loss.detach(),
            "rssm_reward_loss": reward_loss.detach(),
            "rssm_kl": kl.mean().detach(),
        }
        return loss, stats

    @staticmethod
    def _stack_states(states):
        stacked = {k: torch.stack([state[k] for state in states], dim=0) for k in states[0].keys()}
        stacked["dist"] = Independent(Normal(stacked["mean"], stacked["std"]), 1)
        return stacked
