import asyncio
from typing import Any, Awaitable

import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from poke_env.battle import DoubleBattle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

BATTLE_FORMAT = "gen9championsvgc2026regma"
N_FEATURES = 30

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


class PolicyPlayer(Player):
    policy = None

    def __init__(self, policy=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle: DoubleBattle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()

        obs = self.embed_battle(battle)

        obs_dict = {
            "observation": np.array(obs, dtype=np.float32),
        }

        mask = get_action_masks(self.env, self)

        action, _ = self.policy.predict(
            obs_dict,
            action_masks=mask,
            deterministic=False,
        )

        return DoublesEnv.action_to_order(int(action), battle)

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

        fainted_mon_team = len([m for m in battle.team.values() if m.fainted]) / 4
        fainted_mon_opponent = (
            len([m for m in battle.opponent_team.values() if m.fainted]) / 4
        )

        our_hp = np.array(
            [x.current_hp_fraction if x else 0.0 for x in battle.active_pokemon],
            dtype=np.float32,
        )

        opp_hp = np.array(
            [
                x.current_hp_fraction if x else 0.0
                for x in battle.opponent_active_pokemon
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                np.array([fainted_mon_team, fainted_mon_opponent], dtype=np.float32),
                our_hp,
                opp_hp,
            ],
            dtype=np.float32,
        )


class ExampleEnv(DoublesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_spaces = {
            self.possible_agents[0]: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
        }

    @classmethod
    def create_env(cls):
        env = cls(
            battle_format=BATTLE_FORMAT,
            team=TEAM,
            log_level=40,
            open_timeout=None,
        )

        opponent = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=TEAM,
            start_listening=False,
        )

        env = SingleAgentWrapper(env, opponent)

        env = ActionMasker(env, lambda e: DoublesEnv.get_action_mask(e.env.battle1))

        return Monitor(env)

    def calc_reward(self, battle):
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: DoubleBattle):
        return PolicyPlayer.embed_battle(battle)


def train():
    env = ExampleEnv.create_env()

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=98_304, progress_bar=True)

    env.close()

    agent = PolicyPlayer(
        policy=model,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        team=TEAM,
    )

    opponents = [
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
