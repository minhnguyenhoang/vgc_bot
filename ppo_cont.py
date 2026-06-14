import asyncio
from itertools import combinations

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

BATTLE_FORMAT = "gen9championsvgc2026regma"
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


# -----------------------------
# Heuristic RL Player
# -----------------------------
class HeuristicDoublesPlayer(Player):
    def __init__(self, *args, lr=0.01, **kwargs):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.w = None
        self.memory = []

    # -----------------------------
    # STATE
    # -----------------------------
    #
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

    def embed_battle(self, battle: DoubleBattle):
        moves_base_power = -np.ones(8)
        moves_dmg_multiplier = np.ones(16)

        flat_moves = battle.available_moves[0] + battle.available_moves[1]

        for i, move in enumerate(flat_moves[:8]):
            moves_base_power[i] = move.base_power / 100

            for j in range(2):
                opp = battle.opponent_active_pokemon[j]
                if opp is not None:
                    moves_dmg_multiplier[i * 2 + j] = move.type.damage_multiplier(
                        opp.type_1,
                        opp.type_2,
                        type_chart=GenData.from_gen(battle.gen).type_chart,
                    )

        fainted_team = len([p for p in battle.team.values() if p.fainted]) / 4
        fainted_opp = len([p for p in battle.opponent_team.values() if p.fainted]) / 4

        our_hp = np.array(
            [p.current_hp_fraction if p else 0 for p in battle.active_pokemon]
        )
        opp_hp = np.array(
            [p.current_hp_fraction if p else 0 for p in battle.opponent_active_pokemon]
        )

        can_mega = np.array(battle.can_mega_evolve, dtype=np.float32)

        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_team, fainted_opp],
                our_hp,
                opp_hp,
                can_mega,
            ],
            dtype=np.float32,
        )

    # -----------------------------
    # FIXED ACTION ENCODING (IMPORTANT)
    # -----------------------------
    def encode_action(self, a):
        # PASS / DEFAULT / FORFEIT
        if isinstance(a, PassBattleOrder):
            return np.array([0.0, 1.0, 0.0])

        if isinstance(a, DefaultBattleOrder):
            return np.array([0.0, 1.0, 0.0])

        if isinstance(a, ForfeitBattleOrder):
            return np.array([-1.0, 0.0, 0.0])

        # SWITCH (poke-env uses pokemon attribute)
        if hasattr(a, "pokemon") and a.pokemon is not None:
            return np.array([1.0, 0.0, 0.0])

        # MOVE
        if hasattr(a, "move") and a.move is not None:
            bp = getattr(a.move, "base_power", 0) or 0
            return np.array([0.0, 0.0, bp / 100.0])

        return np.array([0.0, 0.0, 0.0])

    def action_features(self, a1, a2):
        return np.concatenate([self.encode_action(a1), self.encode_action(a2)])

    # -----------------------------
    # FEATURES
    # -----------------------------
    def features(self, battle, action):
        a1, a2 = action
        return np.concatenate([self.embed_battle(battle), self.action_features(a1, a2)])

    # -----------------------------
    # SCORE
    # -----------------------------
    def score(self, battle, action):
        f = self.features(battle, action)

        if self.w is None:
            self.w = np.zeros_like(f)

        return float(np.dot(self.w, f))

    # -----------------------------
    # POLICY
    # -----------------------------
    def choose_move(self, battle: DoubleBattle):
        if battle.wait:
            return self.choose_default_move(battle)

        joint_actions = [
            (o1, o2)
            for o1 in battle.valid_orders[0]
            for o2 in battle.valid_orders[1]
            if DoubleBattleOrder.join_orders([o1], [o2])
        ]

        if not joint_actions:
            return self.choose_random_move(battle)

        best_action = None
        best_score = -1e9

        for a1, a2 in joint_actions:
            s = self.score(battle, (a1, a2))
            if s > best_score:
                best_score = s
                best_action = (a1, a2)

        # store FEATURES (NOT battle object)
        self.memory.append(self.features(battle, best_action))

        return DoubleBattleOrder(best_action[0], best_action[1])

    # -----------------------------
    # LEARNING
    # -----------------------------
    def update_after_battle(self, won: bool):
        if self.w is None or not self.memory:
            self.memory = []
            return

        reward = 1.0 if won else -1.0

        for f in self.memory:
            self.w += self.lr * reward * f / len(self.memory)

        self.memory = []

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self.update_after_battle(battle.won)


async def main():
    agent = HeuristicDoublesPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        team=TEAM,
        save_replays=True,
    )

    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT, max_concurrent_battles=1, team=TEAM
    )

    await agent.battle_against(opponent, n_battles=1)

    print("Finished battles:", agent.n_finished_battles)
    print("Wins:", agent.n_won_battles)


if __name__ == "__main__":
    asyncio.run(main())
