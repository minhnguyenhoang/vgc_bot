import asyncio
from typing import List, Tuple

from poke_env import cross_evaluate
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.battle.battle import Battle
from poke_env.battle.effect import Effect
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.target import Target
from poke_env.data import GenData
from poke_env.player import (
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.baselines import PseudoBattle
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from tabulate import tabulate

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


class SHP(Player):
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealthrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    SPEED_TIER_COEFICIENT = 0.75
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -0.25

    @staticmethod
    def _estimate_matchup(mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += SimpleHeuristicsPlayer.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= SimpleHeuristicsPlayer.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * SimpleHeuristicsPlayer.HP_FRACTION_COEFICIENT
        score -= (
            opponent.current_hp_fraction * SimpleHeuristicsPlayer.HP_FRACTION_COEFICIENT
        )

        return score

    @staticmethod
    def _should_mega_evolve(battle: AbstractBattle, n_remaining_mons: int):
        if battle.can_mega_evolve:
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            if (
                SimpleHeuristicsPlayer._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    @staticmethod
    def _should_switch_out(battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if SimpleHeuristicsPlayer._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                SimpleHeuristicsPlayer._estimate_matchup(active, opponent)
                < SimpleHeuristicsPlayer.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        return False

    @staticmethod
    def _stat_estimation(mon: Pokemon, stat: str):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    @staticmethod
    def choose_singles_move(battle: Battle) -> Tuple[SingleBattleOrder, float]:
        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return Player.choose_random_singles_move(battle), 0

        # Rough estimation of damage ratio
        physical_ratio = SimpleHeuristicsPlayer._stat_estimation(
            active, "atk"
        ) / SimpleHeuristicsPlayer._stat_estimation(opponent, "def")
        special_ratio = SimpleHeuristicsPlayer._stat_estimation(
            active, "spa"
        ) / SimpleHeuristicsPlayer._stat_estimation(opponent, "spd")

        if battle.available_moves and (
            not SimpleHeuristicsPlayer._should_switch_out(battle)
            or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.opponent_team.values() if m.fainted is True]
            )

            # Entry hazard...
            for move in battle.available_moves:
                # ...setup
                if (
                    n_opp_remaining_mons >= 3
                    and move.id in SimpleHeuristicsPlayer.ENTRY_HAZARDS
                    and SimpleHeuristicsPlayer.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    return Player.create_order(move), 0

                # ...removal
                elif (
                    battle.side_conditions
                    and move.id in SimpleHeuristicsPlayer.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    return Player.create_order(move), 0

            # Setup moves
            if (
                active.current_hp_fraction == 1
                and SimpleHeuristicsPlayer._estimate_matchup(active, opponent) > 0
            ):
                for move in battle.available_moves:
                    if (
                        move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        return Player.create_order(move), 0

            move, score = max(
                [
                    (
                        m,
                        m.base_power
                        * (1.5 if m.type in active.types else 1)
                        * (
                            physical_ratio
                            if m.category == MoveCategory.PHYSICAL
                            else special_ratio
                        )
                        * m.accuracy
                        * m.expected_hits
                        * opponent.damage_multiplier(m),
                    )
                    for m in battle.available_moves
                ],
                key=lambda x: x[1],
            )
            return (
                Player.create_order(
                    move,
                    mega=SHP._should_mega_evolve(battle, n_remaining_mons),
                ),
                score,
            )

        if battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            return (
                Player.create_order(
                    max(
                        switches,
                        key=lambda s: SimpleHeuristicsPlayer._estimate_matchup(
                            s, opponent
                        ),
                    )
                ),
                0,
            )

        return Player.choose_random_singles_move(battle), 0

    @staticmethod
    def get_double_target_multiplier(battle: DoubleBattle, order: SingleBattleOrder):
        can_target_first_opponent = (
            battle.opponent_active_pokemon[0]
            and not battle.opponent_active_pokemon[0].fainted
        )
        can_target_second_opponent = (
            battle.opponent_active_pokemon[1]
            and not battle.opponent_active_pokemon[1].fainted
        )
        can_double_target = can_target_first_opponent and can_target_second_opponent
        return (
            1
            if not hasattr(order, "order")
            or not isinstance(order.order, Move)
            or order.order.target in {Target.NORMAL, Target.ANY}
            or not can_double_target
            else 1.5
        )

    def choose_move(self, battle: AbstractBattle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_singles_move(battle)[0]  # type: ignore
        orders: List[SingleBattleOrder] = []
        for active_id in [0, 1]:
            mon = battle.active_pokemon[active_id]
            if mon is not None and Effect.COMMANDER in mon.effects:
                orders += [PassBattleOrder()]
                continue
            if mon is None and not battle.available_switches[active_id]:
                orders += [PassBattleOrder()]
                continue
            results = [
                self.choose_singles_move(PseudoBattle(battle, active_id, opp_id))
                for opp_id in [0, 1]
            ]
            possible_orders = [r[0] for r in results]
            scores = [r[1] for r in results]
            for order in possible_orders:
                if (
                    order is not None
                    and hasattr(order, "order")
                    and isinstance(order.order, Move)
                    and mon is not None
                ):
                    target = [o for o in possible_orders].index(order) + 1
                    possible_targets = battle.get_possible_showdown_targets(
                        order.order, mon
                    )
                    if target not in possible_targets:
                        target = possible_targets[0]
                    order.move_target = target
            scores = [
                scores[i]
                * self.get_double_target_multiplier(battle, possible_orders[i])
                for i in [0, 1]
            ]
            orders += [
                (
                    max(results, key=lambda a: a[1])[0]
                    if battle.force_switch != [[False, True], [True, False]][active_id]
                    and not (
                        len(battle.available_switches[active_id]) == 1
                        and battle.force_switch == [True, True]
                        and active_id == 1
                    )
                    else PassBattleOrder()
                )
            ]
        joined_orders = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
        if joined_orders:
            return joined_orders[0]
        else:
            return DoubleBattleOrder(orders[0], DefaultBattleOrder())


async def main():
    player_1 = SHP(
        max_concurrent_battles=75,
        battle_format=BATTLE_FORMAT,
        save_replays="replays",
        team=TEAM,
    )

    players = [player_1] + [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=25, team=TEAM)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]

    cross_evaluation = await cross_evaluate(players, n_challenges=2500)

    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    with open("results/simple_heuristic_player_w_mega.txt", "w", encoding="utf-8") as f:
        f.write(tabulate(table))

    print(tabulate(table))


if __name__ == "__main__":
    asyncio.run(main())
