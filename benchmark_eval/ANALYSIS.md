# GTBench & TextArena 评测分析

测试 4 个 checkpoint 在两个外部博弈 benchmark 上的表现(2026-06-13):

| 模型 | 路径 |
|:-----|:-----|
| Base | `Qwen/Qwen2.5-3B-Instruct` |
| SFT (CoT) | `results/phase2/sft_cot/checkpoint-best` |
| GRPO (verl) | `results/phase2/grpo_verl/merged_step_280` |
| GRPO+Probe (verl) | `results/phase2/grpo_verl_probe/merged_step_320` |

四个模型均在 GameSolve-Bench(矩阵博弈求解)上训练,本评测衡量**训练能力向 sequential / interactive 博弈的迁移**(结构泛化)。

## 评测设置

- **GTBench**(10 游戏,openspiel/rlcard 引擎):候选模型(prompt_agent)对 **随机对手**,每对 20 场有效对局,前后手各半。完成率低的 (model, game) 补跑至多 40 场。原 langchain 后端重写为 OpenAI client 直连本地 vLLM(`gamingbench/chat/chat.py`)。
- **TextArena**(6 游戏:TicTacToe、ConnectFour、Nim、KuhnPoker、IteratedPrisonersDilemma、SimpleNegotiation):候选模型对 **固定 Base 对手**,每对 20 集,前后手交替。Base vs Base 为对照(期望 ~50%)。
- 4 个 3B 模型由 4 个 vLLM 服务器(GPU 0-3)并行供给;GTBench 4 模型 × 8 worker 并行,TextArena 24 任务并行。

完整数表见 `results/REPORT.md` 与 `results/summary.json`。

## 主要结果

**GTBench avg score(8 个有效游戏,排除 nim/pig)**:

SFT **60%** > GRPO+Probe **57%** ≈ GRPO **56%** > Base **52%**

**TextArena avg score(vs Base 对手)**:

GRPO **56%** > Base 对照 **52%** > GRPO+Probe **48%** > SFT **38%**

## 发现

1. **SFT 的迁移表现分裂,整体最差**。GTBench 对随机对手时 SFT 最高(60%,主要靠 prisoners_dilemma 90% 和格式遵循好),但 TextArena 对 Base 对手时**大幅退化至 38%**(KuhnPoker 20%、TicTacToe 35%)。SFT 过拟合 GameSolve 的 CoT 答题格式,在需要多轮交互、对手建模的场景中明显受损。这与此前 OOD 评测结论(SFT 同分布最强、RL 泛化更稳)方向一致,且在更远的分布外进一步放大。

2. **RL 方法(GRPO / GRPO+Probe)不损害通用博弈能力**。两者在 GTBench(56%/57%)和 TextArena(56%/48%)上与 Base(52%/52%)持平或略升。在 GameSolve 上 +25pp 的提升(0.77 vs 0.52)没有以牺牲交互式博弈能力为代价——RL 微调的"能力税"低于 SFT。

3. **Probe 与 GRPO 在外部 benchmark 上互有胜负,无一致差异**。GTBench 上 probe 略高(57% vs 56%),TextArena 上略低(48% vs 56%,差异集中在 Nim 30%-vs-60% 与 TicTacToe)。Probe 的概念表征(均衡类型、支配策略等)是矩阵博弈特定的,不期望迁移到 sequential 游戏;结果与该预期一致。

4. **格式遵循是 3B 模型在 GTBench 的主要瓶颈**(独立发现)。pig 游戏(动作仅 `<roll>`/`<stop>`)中 Base/GRPO/Probe 几乎全部因输出字面 `<Action>` 等非法动作被判 Abnormal(完成率 1-3%),而 SFT 达 77%——SFT 显著提升指令格式遵循。各模型总体完成率 79-83%,失败几乎全为候选模型非法动作。

5. **结构性局限**:prisoners_dilemma 上全部微调模型 > Base(72-90% vs 60%),是唯一所有训练方法一致受益的 GTBench 游戏——它最接近训练分布(收益矩阵推理)。negotiation 全平局(双方都难以成交),无判别力。

## 统计注意事项

每格 20 场对局,95% 置信区间约 ±22pp;单格差异大多不显著,应只解读跨游戏的一致模式(如 SFT 在 TextArena 的全面下滑、RL 方法的稳定性)。pig/nim 因完成率低导致有效样本不一,已在均值中排除或标注 n。

## 结论(对 proposal Phase 2 的含义)

- 结构泛化(matrix → sequential)有限,这是预期内的:probe 概念与 GameSolve 任务绑定。若要外部 benchmark 收益,需在训练数据中混入 sequential 博弈或把概念标签泛化(如"是否存在占优行动"在 sequential 游戏中重新定义)。
- RL(verl GRPO ± probe)是更安全的能力增强路径:同分布大幅提升且分布外无回退;SFT 的同分布优势以分布外退化为代价。
- 后续若做 benchmark 提升实验,优先 TextArena 自我对弈 RL(SPIRAL 式)或 GTBench 游戏混训。

## 复现

```bash
# 1. 启动 4 个 vLLM 服务器(GPU 0-3,端口 8001-8004)
bash scripts/serve_models.sh   # 见 README

# 2. GTBench(模型名 = qwen3b-base|qwen3b-sft|qwen3b-grpo|qwen3b-grpo-probe)
bash scripts/run_gtbench.sh qwen3b-base 20

# 3. TextArena
python3 scripts/textarena_eval.py --model qwen3b-base --episodes 20

# 4. 汇总
python3 scripts/analyze_results.py
```
