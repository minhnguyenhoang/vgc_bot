import asyncio
import os
import shutil
from itertools import combinations
from pprint import pprint
from typing import Any, Awaitable, Optional, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
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
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
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

    # -------------------------
    # MAIN FIX
    # -------------------------
    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)

        B = action_logits.shape[0]
        N = DoublesEnv.get_action_space_size(
            GenData.from_format(BATTLE_FORMAT).gen
        )  # fixed from environment

        # reshape logits into joint grid
        logits = action_logits.view(B, N, N)

        # original mask is still [B, 214]
        mask = self._mask

        mask_p1 = mask[:, :N]
        mask_p2 = mask[:, N:]

        # build joint mask via AND (your rule)
        joint_mask = mask_p1[:, :, None] & mask_p2[:, None, :]

        invalid = torch.ones((N, N), device=logits.device, dtype=torch.int)

        invalid_ranges = [
            (27, 46),
            (47, 66),
            (67, 86),
            (87, 106),
        ]

        # Once-per-battle mechanics
        for l, r in invalid_ranges:
            invalid[l : r + 1, l : r + 1] = 0

        # Invalid switching
        idx = torch.arange(1, 7, device=invalid.device)
        invalid[idx, idx] = 0

        final_mask = joint_mask & invalid

        # apply mask
        logits = logits.masked_fill(final_mask == 0, -1e9)

        # flatten back
        logits = logits.view(B, N * N)

        return self.action_dist.proba_distribution(logits)

    # -------------------------
    # ENSURE MASK IS AVAILABLE
    # -------------------------
    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        actions, values, log_prob = super().forward(obs, deterministic)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        # must match forward masking
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)


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
            agent: Box(-5, 5, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: Discrete(
                DoublesEnv.get_action_space_size(GenData.from_format(BATTLE_FORMAT).gen)
                ** 2
            )
            for agent in self.possible_agents
        }

    @classmethod
    def create_env(cls) -> Monitor:
        env = cls(
            battle_format=BATTLE_FORMAT,
            log_level=20,
            open_timeout=None,
            # team=TEAM,
            save_replays="replays/RL_Training",
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

    @staticmethod
    def action_to_order(
        action: int,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = False,
    ) -> BattleOrder:
        strict = False
        s = DoublesEnv.get_action_space_size(GenData.from_format(BATTLE_FORMAT).gen)
        npaction = np.int64(action)
        a1, a2 = npaction // s, npaction % s
        print(npaction, a1, a2)
        if a1 == -2 and a2 == -2:
            return DefaultBattleOrder()
        elif a1 == -1 or a2 == -1:
            return ForfeitBattleOrder()
        try:
            order1 = RLEnv._action_to_order_individual(a1, battle, fake, 0)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order1 = (
                    order.first_order
                    if not isinstance(order, DefaultBattleOrder)
                    else order
                )
        try:
            order2 = RLEnv._action_to_order_individual(a2, battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order2 = (
                    order.second_order
                    if not isinstance(order, DefaultBattleOrder)
                    else order
                )
        joined_orders = DoubleBattleOrder.join_orders([order1], [order2])
        if not joined_orders:
            error_msg = (
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - converted orders {order1} "
                f"and {order2} are incompatible!"
            )
            if strict:
                raise ValueError(error_msg)
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg + " Defaulting to random move.")
                return Player.choose_random_doubles_move(battle)
        else:
            return joined_orders[0]

    @staticmethod
    def _action_to_order_individual(
        action: np.int64, battle: DoubleBattle, fake: bool, pos: int
    ) -> SingleBattleOrder:
        if action == -2:
            return DefaultBattleOrder()
        elif action == 0:
            order: SingleBattleOrder = PassBattleOrder()
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            if active_mon is None:
                raise ValueError(
                    f"Invalid order from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move, but battle.active_pokemon is None!"
                )
            avail_ids = [m.id for m in battle.available_moves[pos]]
            known_moves = list(active_mon.moves.values())[:4]
            known_ids = [m.id for m in known_moves]
            mvs = (
                battle.available_moves[pos]
                if len(avail_ids) == 1 and avail_ids[0] not in known_ids
                else known_moves
            )
            if (action - 7) % 20 // 5 not in range(len(mvs)):
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move but the move index {(action - 7) % 20 // 5} "
                    f"is out of bounds for available moves {mvs}!"
                )
            order = Player.create_order(
                mvs[(action - 7) % 20 // 5],
                move_target=(action.item() - 7) % 5 - 2,
                mega=(action - 7) // 20 == 1,
                z_move=(action - 7) // 20 == 2,
                dynamax=(action - 7) // 20 == 3,
                terastallize=(action - 7) // 20 == 4,
            )
        if not fake and str(order) not in [str(o) for o in battle.valid_orders[pos]]:
            raise ValueError(
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} at position {pos} - order {order} "
                f"not in action space {[str(o) for o in battle.valid_orders[pos]]}!"
            )
        return order

    @staticmethod
    def order_to_action(
        order: BattleOrder,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = False,
    ) -> np.int64:
        strict = False
        if isinstance(order, DefaultBattleOrder):
            return np.array([-2, -2])
        elif isinstance(order, ForfeitBattleOrder):
            return np.array([-1, -1])
        assert isinstance(order, DoubleBattleOrder)
        joined_orders = DoubleBattleOrder.join_orders(
            [order.first_order], [order.second_order]
        )
        if not fake and not joined_orders:
            error_msg = (
                f"Invalid order {order} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - orders are incompatible!"
            )
            if strict:
                raise ValueError(error_msg)
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg + " Defaulting to random move.")
                return order_to_action(
                    Player.choose_random_doubles_move(battle), battle, fake, strict
                )
        try:
            action1 = DoublesEnv._order_to_action_individual(
                order.first_order, battle, fake, 0
            )
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action1 = DoublesEnv._order_to_action_individual(
                    (
                        order.first_order
                        if not isinstance(order, DefaultBattleOrder)
                        else order
                    ),
                    battle,
                    fake,
                    0,
                )
        try:
            action2 = DoublesEnv._order_to_action_individual(
                order.second_order, battle, fake, 1
            )
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action2 = DoublesEnv._order_to_action_individual(
                    (
                        order.second_order
                        if not isinstance(order, DefaultBattleOrder)
                        else order
                    ),
                    battle,
                    fake,
                    1,
                )
        return np.int64(
            int(action1)
            * DoublesEnv.get_action_space_size(GenData.from_format(BATTLE_FORMAT).gen)
            + int(action2)
        )


def train():
    folder = "replays/RL_Training"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

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

    # Training
    print("Training...")
    ppo.learn(98_304, progress_bar=True)
    env.close()

    # Testing/Evaluation
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
