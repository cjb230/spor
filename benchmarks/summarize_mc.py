"""Summarize saved Monte Carlo experiments without pooling different budgets."""
import collections
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent / "mc-comparison-2026-11-19"
    records = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if "standings" not in data:
            continue
        if data["exit_code"] != 0:
            raise RuntimeError(f"failed experiment: {path}")
        records.append((path.stem, data))
    if len({data["binary_sha256"] for _, data in records}) != 1:
        raise RuntimeError("matchups used different executables")
    expected = {f"s8_block{block}_{order}" for block in [1, 2, 3] for order in ["forward", "reverse"]}
    expected |= {f"{condition}_{order}" for condition in ["s16_block1", "budget256"] for order in ["forward", "reverse"]}
    if {name for name, _ in records} != expected:
        raise RuntimeError(f"incomplete experiment set; missing {sorted(expected - {name for name, _ in records})}")
    lines = ["# Monte Carlo comparison results", "",
             "M4 Pro, 14 tournament workers. Commands and raw output are saved per experiment.", "",
             "| Run | Deck start | MCB wins | MCP wins | Draws | Wall seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    totals = collections.defaultdict(lambda: [0, 0, 0])
    for name, data in records:
        wins = {row["strategy"].split(":")[0]: row["wins"] for row in data["standings"]}
        draws = data["standings"][0]["draws"]
        lines.append(f"| {name} | {data['deck_seed']} | {wins['mcb']} | {wins['mcp']} | {draws} | {data['elapsed_seconds']:.2f} |")
        group = "eight samples" if name.startswith("s8_") else "sixteen samples" if name.startswith("s16_") else "approximate matched compute"
        for index, count in enumerate([wins["mcb"], wins["mcp"], draws]):
            totals[group][index] += count
    lines.extend(["", "## Totals by condition", ""])
    for group, (mcb, mcp, draws) in totals.items():
        total = mcb + mcp + draws
        lines.append(f"- {group}: MCB {mcb}, MCP {mcp}, draws {draws}; MCB win rate {100 * mcb / total:.1f}% over {total} games.")
    costs = json.loads((root / "rollout-cost.json").read_text())["records"]
    pairs = collections.defaultdict(dict)
    for row in costs:
        pairs[row["deck_seed"], row["turn"], row["repeat"]][row["policy"]] = row["seconds"]
    ratios = [row["Planner"] / row["Builder"] for row in pairs.values()]
    aggregate = sum(row["Planner"] for row in pairs.values()) / sum(row["Builder"] for row in pairs.values())
    lines.extend(["", "## Identical-position cost", "",
                  f"18 paired decisions on six positions: median MCP/MCB time ratio {statistics.median(ratios):.2f}x; aggregate ratio {aggregate:.2f}x.",
                  "Both use eight samples, identical observations/world seeds, and one decision worker. No matches ran concurrently with this probe.", "",
                  "## Interpretation limits", "",
                  "Mirrored games share decks and are paired, so pooled games should not be treated as independent Bernoulli observations for a simple binomial confidence interval.",
                  "The 20-game pilot overlaps block 1 and is excluded. The user's earlier 579–421 result is external reference data, not pooled here.",
                  "MCB:256 versus MCP:8 approximates equal compute using the measured ratio; it does not enforce identical elapsed time per move.",
                  "No bot policies or engine code were changed. The exact matchup executable SHA-256 is stored in each report.", ""])
    text = "\n".join(lines)
    (root / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
