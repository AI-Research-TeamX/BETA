# Benchmark Evaluation: GTBench & TextArena

4 个 checkpoint:Base (Qwen2.5-3B-Instruct)、SFT (CoT)、GRPO (verl, step280)、GRPO+Probe (verl, step320)。

## GTBench — LLM (prompt agent) vs Random agent

指标:score = (win + 0.5*draw)/normal;`n` 为有效(Normal)对局数。
avg score 仅在所有模型 n≥10 的游戏上计算(排除:nim, pig)。

| Game | Base (Qwen2.5-3B) | SFT (CoT) | GRPO (verl) | GRPO+Probe (verl) |
|:-----|----:|----:|----:|----:|
| breakthrough | 55% (n=20) | 55% (n=20) | 50% (n=20) | 50% (n=20) |
| connect4 | 70% (n=20) | 60% (n=20) | 45% (n=20) | 65% (n=20) |
| first_sealed_auction | 30% (n=20) | 60% (n=20) | 55% (n=20) | 36% (n=11) |
| kuhn_poker | 70% (n=20) | 55% (n=20) | 75% (n=20) | 70% (n=20) |
| liars_dice | 50% (n=20) | 53% (n=30) | 45% (n=20) | 60% (n=20) |
| negotiation | 50% (n=20) | 50% (n=20) | 50% (n=20) | 50% (n=20) |
| nim | 42% (n=19) | 86% (n=7) | 15% (n=13) | 33% (n=54) |
| pig | 0% (n=1) | 20% (n=20) | — | 100% (n=2) |
| prisoners_dilemma | 60% (n=20) | 90% (n=20) | 72% (n=20) | 80% (n=20) |
| tictactoe | 34% (n=22) | 60% (n=20) | 55% (n=20) | 48% (n=20) |
| **avg score** | 52% | 60% | 56% | 57% |
| **avg completion** | 80% | 83% | 79% | 79% |

## TextArena — candidate vs fixed Base opponent

指标:score = (win + 0.5*draw)/valid。Base vs Base 行为对照(期望 ~50%)。

| Env | Base (Qwen2.5-3B) | SFT (CoT) | GRPO (verl) | GRPO+Probe (verl) |
|:----|----:|----:|----:|----:|
| ConnectFour-v0 | 45% | 37% | 60% | 45% |
| IteratedPrisonersDilemma-v0 | 50% | 45% | 45% | 50% |
| KuhnPoker-v0 | 70% | 20% | 55% | 65% |
| Nim-v0 | 35% | 35% | 60% | 30% |
| SimpleNegotiation-v0 | 42% | 57% | 57% | 65% |
| TicTacToe-v0 | 70% | 35% | 57% | 35% |
| **avg score** | 52% | 38% | 56% | 48% |
