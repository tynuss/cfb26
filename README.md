# cfb26
2026 college football spread model, published at https://tynuss.github.io/cfb26/

- `index.html` is the page: one self-contained file with the week's model lines, DraftKings lines, edges and Strong/Lean/Pass signals built in, so it also works offline. Online, this week's games show live ESPN scores, and the week menu has every past week of the season with final scores and whether the model's pick covered the closing line.
- `cfb_weekly.py` builds it from public data (sportsdataverse, ESPN odds and logos). The model line is the DraftKings line adjusted by what the model's inputs say the market tends to miss; the adjustment weights were fit on 2008–2025 games, last 4 seasons counted fully. A Lean is a 2+ point difference (2010–2025 backtest, each season fit only on earlier ones: 1,065–912, 53.9%).
- `lines/<season>.json` keeps every DraftKings line pulled this season, so games that have already kicked off keep their pre-game line when the page is rebuilt mid-week.
- `.github/workflows/weekly.yml` runs the model every Monday around 11:40 AM CT and commits the new `index.html`; GitHub Pages republishes it a minute or two later. To rebuild early, open the Actions tab, pick "Weekly model update" and press "Run workflow".
- `fan_sentiment.py` scores each team's fan attitude from the week's Google News headlines (local papers, beat writers, fan sites) into `sentiment/<season>.csv`. It is shown on the page and tracked, not used in the model lines. The "Fan attitude history" workflow scores past seasons for backtesting.
- Optional `page_inputs.json` adds team adjustments (committing it rebuilds the page), e.g. `{"overrides": [{"team_id": "333", "team": "Alabama", "points": -3, "note": "QB out"}]}`.
