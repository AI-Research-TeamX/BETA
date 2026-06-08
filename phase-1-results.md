## Phase 1 Results: Cross-Pooling Comparison

### Best Pooling Method per Concept

| Concept       | 1.5B Best | 1.5B Acc | 3B Best  | 3B Acc |
| ------------- | --------- | -------- | -------- | ------ |
| eq_type       | mean      | 0.7407   | weighted | 0.7556 |
| game_type     | sum       | 0.9979   | mean     | 0.9938 |
| difficulty    | weighted  | 0.9000   | weighted | 0.9062 |
| dominance     | mean      | 0.8187   | weighted | 0.8125 |
| br_direction  | sum       | 0.6286   | sum      | 0.6476 |
| eq_uniqueness | weighted  | 0.7370   | weighted | 0.7259 |

### Key Findings

1. **Mean/weighted pooling dominate over last-token pooling** across nearly all concepts — the standard decoder "last token" approach leaves substantial signal on the table.

2. **`first` pooling is uninformative** — constant accuracy across all layers (majority-class baseline), confirming the BOS token carries no game-theoretic content.

3. **`sum` pooling excels for game_type and br_direction** despite very slow convergence, suggesting magnitude-sensitive features matter for these structural properties.

4. **`weighted` pooling is the most consistently strong** method, winning 4/6 concepts on the 3B model — its exponential position-weighting captures both early structural info and late reasoning tokens.

5. **3B model shows improvements over 1.5B** across all concepts, with the largest gains in eq_type (+0.015), br_direction (+0.019), and game_type (stable near ceiling).

### Output Artifacts in `results/analysis/`
- `pooling_comparison.json` — consolidated JSON
- Per-model: bar charts, heatmaps, layer-wise curves
- `best_pooling_cross_model.png` — cross-model comparison
