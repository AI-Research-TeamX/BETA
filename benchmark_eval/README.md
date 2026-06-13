# benchmark_eval — GTBench & TextArena 外部基准评测

评测 4 个 checkpoint(Base / SFT / GRPO(verl) / GRPO+Probe(verl))在外部博弈 benchmark 上的迁移表现。

**结论速览**:RL 方法(GRPO±Probe)在分布外博弈上与 Base 持平或略升,SFT 在 TextArena 上显著退化(38% vs Base 52%);Probe 与 GRPO 在外部 benchmark 上无一致差异。详见 [ANALYSIS.md](ANALYSIS.md)。

## 目录结构

```
benchmark_eval/
├── GTBench/            # github.com/jinhaoduan/GTBench 克隆
│   └── gamingbench/chat/chat.py   # [改] langchain → openai client 直连本地 vLLM
│   └── gamingbench/configs/model_configs/qwen3b-*.yaml  # [增] 4 个模型配置
├── TextArena/          # github.com/LeonGuertler/TextArena 克隆(pip install -e)
├── scripts/
│   ├── serve_models.sh       # 启动 4 个 vLLM 服务器(GPU 0-3,端口 8001-8004)
│   ├── run_gtbench.sh        # GTBench:<model> vs random,用法见文件头
│   ├── textarena_eval.py     # TextArena:<model> vs 固定 base 对手
│   └── analyze_results.py    # 汇总两个 benchmark → REPORT.md + summary.json
├── results/
│   ├── gtbench/<model>/<game>/*.jsonl   # 每场对局完整记录
│   ├── textarena/<model>__<env>.json    # 每 (模型,游戏) 20 集
│   ├── REPORT.md / summary.json         # 汇总表
├── logs/               # vLLM 服务器与各评测任务日志
├── ANALYSIS.md         # 结果分析(主要发现、统计注意事项、结论)
└── README.md
```

## 运行

```bash
bash scripts/serve_models.sh                      # 1. 起服务器
for M in qwen3b-base qwen3b-sft qwen3b-grpo qwen3b-grpo-probe; do
  bash scripts/run_gtbench.sh $M 20 &             # 2. GTBench(可并行)
  python3 scripts/textarena_eval.py --model $M &  # 3. TextArena(可并行)
done; wait
python3 scripts/analyze_results.py                # 4. 汇总
pkill -f "vllm serve"                             # 5. 释放 GPU
```

依赖:`pip install open_spiel rlcard python-box ml_collections jsonlines && pip install -e TextArena`

## 注意事项

- 本地 vLLM 端点会绕过用户代理(脚本内设置 `NO_PROXY`);若改动端口,需同步 `run_gtbench.sh` 的 `LOCAL_LLM_ENDPOINTS` 与 `textarena_eval.py` 的 `ENDPOINTS`。
- GTBench 对局因候选模型非法动作判 Abnormal 不计入胜率;`analyze_results.py` 对 n<10 的格子在均值中排除(pig/nim)。
- TextArena 候选与对手前后手交替,Base vs Base 为 ~50% 对照。

## GTBench/TextArena 克隆不入库

两个 benchmark 为独立 git 克隆,不纳入本仓库。重新搭建时:
```bash
git clone https://github.com/jinhaoduan/GTBench benchmark_eval/GTBench
git clone https://github.com/LeonGuertler/TextArena benchmark_eval/TextArena
cp scripts/gtbench_patch/chat.py benchmark_eval/GTBench/gamingbench/chat/chat.py
cp scripts/gtbench_patch/model_configs/*.yaml benchmark_eval/GTBench/gamingbench/configs/model_configs/
```
