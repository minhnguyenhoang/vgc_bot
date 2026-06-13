import asyncio

from poke_env.player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate

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


async def main():
    player_1 = RandomPlayer(
        max_concurrent_battles=25,
        team=TEAM,
        battle_format="gen9championsvgc2026regma",
        save_replays="replays",
        start_timer_on_battle_start=True,
    )
    player_2 = RandomPlayer(
        max_concurrent_battles=25, team=TEAM, battle_format="gen9championsvgc2026regma"
    )

    players = [player_1, player_2]

    cross_evaluation = await cross_evaluate(players, n_challenges=100)

    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    print(tabulate(table))


if __name__ == "__main__":
    asyncio.run(main())
