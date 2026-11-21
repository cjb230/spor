# Monte Carlo comparison results

M4 Pro, 14 tournament workers. Commands and raw output are saved per experiment.

| Run | Deck start | MCB wins | MCP wins | Draws | Wall seconds |
|---|---:|---:|---:|---:|---:|
| budget256_forward | 1001 | 47 | 3 | 0 | 148.48 |
| budget256_reverse | 1001 | 49 | 1 | 0 | 150.13 |
| s16_block1_forward | 1001 | 61 | 39 | 0 | 295.58 |
| s16_block1_reverse | 1001 | 50 | 50 | 0 | 295.60 |
| s8_block1_forward | 1001 | 64 | 36 | 0 | 136.28 |
| s8_block1_reverse | 1001 | 63 | 37 | 0 | 137.70 |
| s8_block2_forward | 2001 | 58 | 42 | 0 | 139.30 |
| s8_block2_reverse | 2001 | 60 | 40 | 0 | 138.28 |
| s8_block3_forward | 3001 | 57 | 43 | 0 | 142.44 |
| s8_block3_reverse | 3001 | 61 | 39 | 0 | 137.87 |

## Totals by condition

- approximate matched compute: MCB 96, MCP 4, draws 0; MCB win rate 96.0% over 100 games.
- sixteen samples: MCB 111, MCP 89, draws 0; MCB win rate 55.5% over 200 games.
- eight samples: MCB 363, MCP 237, draws 0; MCB win rate 60.5% over 600 games.

## Identical-position cost

18 paired decisions on six positions: median MCP/MCB time ratio 27.46x; aggregate ratio 31.70x.
Both use eight samples, identical observations/world seeds, and one decision worker. No matches ran concurrently with this probe.

## Interpretation limits

Mirrored games share decks and are paired, so pooled games should not be treated as independent Bernoulli observations for a simple binomial confidence interval.
The 20-game pilot overlaps block 1 and is excluded. The user's earlier 579–421 result is external reference data, not pooled here.
MCB:256 versus MCP:8 approximates equal compute using the measured ratio; it does not enforce identical elapsed time per move.
No bot policies or engine code were changed. The exact matchup executable SHA-256 is stored in each report.
