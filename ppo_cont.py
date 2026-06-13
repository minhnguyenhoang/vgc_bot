import asyncio
from itertools import combinations

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder

FEATURE_DIM = 32
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


class HeuristicRLPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w = None

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

    def choose_move(self, battle: DoubleBattle):
        # Chooses a move with the highest base power when possible
        if battle.available_moves:
            # Iterating over available moves to find the one with the highest base power
            joint_actions = [
                (o1, o2)
                for o1 in battle.valid_orders[0]
                for o2 in battle.valid_orders[1]
                if DoubleBattleOrder.join_orders([o1], [o2])
            ]

            return DoubleBattleOrder(joint_actions[0][0], joint_actions[0][1])
        else:
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)

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


async def main():
    player_1 = HeuristicRLPlayer(
        max_concurrent_battles=1,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
        save_replays="replays",
    )
    player_2 = RandomPlayer(
        max_concurrent_battles=1, battle_format=BATTLE_FORMAT, team=TEAM
    )

    await player_1.battle_against(player_2, n_battles=1)

    print(f"Finished battles: {player_1.n_finished_battles}")
    print(f"Player 1 wins: {player_1.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())
