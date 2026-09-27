from collections import defaultdict

import torch
import torch.optim as optim

from .ppo import PPO
from .wasabi import WasabiAlgoMixin


class RSSMPPO(PPO):
    def __init__(
        self,
        *args,
        rssm_learning_rate: float = 1.0e-3,
        rssm_weight_decay: float = 0.0,
        rssm_grad_clip: float = 100.0,
        rssm_num_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not hasattr(self.actor_critic, "rssm"):
            raise ValueError("RSSMPPO requires a policy with an `rssm` module, e.g. RSSMEncoderMoEActorCritic.")
        self.rssm_grad_clip = rssm_grad_clip
        self.rssm_num_epochs = rssm_num_epochs
        self.rssm_optimizer = optim.AdamW(
            self.actor_critic.rssm.parameters(),
            lr=rssm_learning_rate,
            weight_decay=rssm_weight_decay,
        )

    def act(self, obs, critic_obs):
        actions = super().act(obs, critic_obs)
        if self.transition.rssm_features is None:
            raise RuntimeError("RSSM policy did not produce rssm_features during rollout.")
        return actions

    def _precompute_rssm_features(self):
        with torch.no_grad():
            posts, _ = self.actor_critic.rssm.rollout_features(
                self.storage.observations,
                self.storage.actions,
                self.storage.dones,
            )
            self.storage.rssm_features = posts["deter"].detach()

    def _update_rssm(self):
        mean_stats = defaultdict(float)
        for _ in range(self.rssm_num_epochs):
            loss, stats = self.actor_critic.rssm.compute_loss(
                self.storage.observations,
                self.storage.actions,
                self.storage.rewards,
                self.storage.dones,
            )
            self.rssm_optimizer.zero_grad()
            loss.backward()
            if self.rssm_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_critic.rssm.parameters(), self.rssm_grad_clip)
            self.rssm_optimizer.step()
            mean_stats["rssm_loss"] += loss.detach()
            for key, value in stats.items():
                mean_stats[key] += value
        for key in mean_stats:
            mean_stats[key] = mean_stats[key] / self.rssm_num_epochs
        return mean_stats

    def update(self, current_learning_iteration):
        rssm_stats = self._update_rssm()
        self._precompute_rssm_features()
        mean_losses, average_stats = super().update(current_learning_iteration)
        mean_losses.update(rssm_stats)
        return mean_losses, average_stats

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["rssm_optimizer"] = self.rssm_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "rssm_optimizer" in state_dict:
            self.rssm_optimizer.load_state_dict(state_dict["rssm_optimizer"])


class RSSMWasabiPPO(WasabiAlgoMixin, RSSMPPO):
    pass
