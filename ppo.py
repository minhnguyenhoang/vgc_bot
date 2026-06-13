import asyncio
from itertools import combinations
from pprint import pprint
from typing import Any, Awaitable, Optional, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.data import GenData
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.battle_order import DefaultBattleOrder
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

N_FEATURES = 32
BATTLE_FORMAT = "gen9randomdoublesbattle"
TEAM = """Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 8 Def / 10 SpD / 16 Spe
Careful Nature
- Flare Blitz
- Fake Out
- Parting Shot
- Throat Chop

Sinistcha-Masterpiece @ Kasib Berry
Ability: Hospitality
Level: 50
EVs: 32 HP / 7 Def / 27 SpD
Relaxed Nature
- Matcha Gotcha
- Rage Powder
- Trick Room
- Protect

Floette-Mega (F) @ Floettite
Ability: Fairy Aura
Level: 50
EVs: 18 HP / 1 Def / 15 SpA / 32 Spe
Modest Nature
- Moonblast
- Dazzling Gleam
- Calm Mind
- Protect

Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
EVs: 4 HP / 30 Atk / 32 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Dragon Claw
- Stomping Tantrum

Charizard-Mega-Y @ Charizardite Y
Ability: Drought
Level: 50
EVs: 15 HP / 18 Def / 1 SpA / 32 Spe
Modest Nature
- Heat Wave
- Solar Beam
- Weather Ball
- Protect

Venusaur @ Focus Sash
Ability: Chlorophyll
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Sludge Bomb
- Earth Power
- Sleep Powder
- Protect
"""


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[64, 64],
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        actions, values, log_prob = super().forward(obs, deterministic)

        print("sampled actions:", actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        mask[0][0] = float("-inf")
        mask[0][107] = float("-inf")
        a = self.action_dist.proba_distribution(action_logits + mask)
        print(a, mask)
        for idx, value in enumerate(mask[0]):
            if value != float("-inf"):
                print(idx)
        return a


class FeaturesExtractor(BaseFeaturesExtractor):
    """Extracts the observation tensor from the dict obs and declares features_dim
    so SB3 builds the MLP with the right input size."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class RLPlayer(Player):
    policy: ActorCriticPolicy | None

    def __init__(
        self, policy: ActorCriticPolicy | None = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def teampreview(self, battle):
        # 1 = Incineroar, 2 = Sinistcha, 3 = Mega Floette, 4 = Garchomp, 5 = Mega Charizard Y, 6 = Venusaur
        teams = ["5614", "5624", "5612", "1324", "1326"]

        def teampreview_performance(mon_a, mon_b):
            a_on_b = b_on_a = -np.inf
            for type_ in mon_a.types:
                if type_:
                    a_on_b = max(
                        a_on_b,
                        type_.damage_multiplier(
                            *mon_b.types, type_chart=GenData.from_gen(9).type_chart
                        ),
                    )
            for type_ in mon_b.types:
                if type_:
                    b_on_a = max(
                        b_on_a,
                        type_.damage_multiplier(
                            *mon_a.types, type_chart=GenData.from_gen(9).type_chart
                        ),
                    )
            return a_on_b - b_on_a

        team_perms = [
            [list(battle.team.values())[int(x) - 1] for x in y] for y in teams
        ]
        opponent_team_perms = list(combinations(battle.opponent_team.values(), 4))

        score = {x: 0 for x in battle.team.values()}
        for mon in battle.team.values():
            perf = []
            for opp_team in opponent_team_perms:
                a = [teampreview_performance(mon, opp) for opp in opp_team]
                perf.append(np.mean(a))
            score[mon] = np.median(perf) * 0.65 + np.mean(perf) * 0.35 + 1

        avg_score = sorted(
            [(sum([score[x] for x in team]), team) for team in team_perms]
        )

        return "/team " + teams[team_perms.index(avg_score[-1][1])]

    def choose_move(self, battle: DoubleBattle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        mask = np.array(DoublesEnv.get_action_mask(battle))
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(
                    obs, device=self.policy.device
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    mask, device=self.policy.device
                ).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)
        action = action.cuda().numpy()[0]
        return DoublesEnv.action_to_order(action, battle, strict=False)

    @staticmethod
    def embed_battle(battle: DoubleBattle):
        moves_base_power = -np.ones(8)
        moves_dmg_multiplier = np.ones(16)
        for i, move in enumerate(battle.available_moves[0] + battle.available_moves[1]):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon[0] is not None:
                moves_dmg_multiplier[i * 2] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon[0].type_1,
                    battle.opponent_active_pokemon[0].type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart,
                )
            if battle.opponent_active_pokemon[1] is not None:
                moves_dmg_multiplier[i * 2 + 1] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon[1].type_1,
                    battle.opponent_active_pokemon[1].type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart,
                )
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 4
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 4
        )
        our_hp = tuple(
            x.current_hp_fraction if x else 0.0 for x in battle.active_pokemon
        )
        opp_hp = tuple(
            x.current_hp_fraction if x else 0.0 for x in battle.opponent_active_pokemon
        )
        can_mega = tuple(1 if x else 0 for x in battle.can_mega_evolve)
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                our_hp,
                opp_hp,
                can_mega,
            ],
            dtype=np.float32,
        )


class RLEnv(DoublesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod
    def create_env(cls) -> Monitor:
        env = cls(
            battle_format=BATTLE_FORMAT,
            log_level=40,
            open_timeout=None,
            # team=TEAM,
            save_replays="replays",
        )
        opponent = SimpleHeuristicsPlayer(start_listening=False)
        return Monitor(SingleAgentWrapper(env, opponent))

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return RLPlayer.embed_battle(battle)


def train():
    # setup
    num_envs = 2
    env = RLEnv.create_env()
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=3072 // num_envs,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda",
    )

    # train
    ppo.learn(98_304, progress_bar=True)
    env.close()

    # evaluate
    agent = RLPlayer(
        policy=ppo.policy,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        team=TEAM,
        save_replays="replays",
    )
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10, team=TEAM)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots ---")
    for opp in opponents:
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        print(f"{opp.username}: {win_rate}%")


if __name__ == "__main__":
    train()
