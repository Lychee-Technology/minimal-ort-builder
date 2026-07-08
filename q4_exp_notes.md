# q4 / q4f16 低精度实验记录 (issue #27)

## TL;DR

`jinaai/jina-embeddings-v5-text-nano-retrieval` 官方的 `q4` / `q4f16` 导出对 fp32 的
pooled cosine 只有 ~0.62(个别样本崩到 0.178),不适合检索。本记录用两个实验查清原因:

1. **自建 4-bit spike**:同一个 fp32 `model.onnx`,用 GPTQ / HQQ / RTN 做 4-bit,量化保真度。
2. **结构对比 inspect**:直接读 jina 各变体和我们输出的量化算子属性。

**结论**:对这个"低冗余 nano + last-token pooling"模型,**4-bit 在两处都踩悬崖**——
compute 权重(84 个 MatMul)和 token embedding 表——`q4f16` 再叠一层 fp16 计算制造个别样本崩溃。
官方 8-bit `quantized`(两处都 8-bit)保持 ~0.99。**决策**:发布用 `quantized`;不发布任何 4-bit
target;pipeline 的 `gptq4`/`q4gptq` 能力保留给其它能容忍 4-bit 的模型。

---

## 实验一:自建 4-bit 量化保真度(`scripts/spike_gptq_4bit.py`)

**方法**:onnxruntime 1.27.0,`MatMulNBitsQuantizer`,`block_size=32`, `bits=4`,量化 fp32
`onnx/model.onnx`;在 3 条 fixture 样本 @128 tokens 上算 pooled last-token(归一化)与 raw
`output[0]` 的 cosine vs fp32。RTN 作为"未标定 4-bit"基线(应复现 jina 的低分做健全性检查)。

**结果**:

| algo | pooled min | pooled mean | raw mean | per-sample pooled | 体积 |
|---|---|---|---|---|---|
| RTN(基线) | 0.631 | 0.793 | 0.833 | 0.631 / 0.828 / 0.919 | 469 MB |
| **GPTQ** | **0.682** | **0.810** | 0.842 | 0.682 / 0.826 / 0.921 | 469 MB |
| HQQ | 0.133 | 0.626 | 0.658 | 0.133 / 0.814 / 0.930 | 481 MB |
| *8-bit `quantized`*(参考) | *~0.988* | *~0.994* | — | — | *~247 MB* |

要点:
- **GPTQ 几乎没救回 4-bit**(mean 0.810 vs RTN 0.793,min 0.682 vs 0.631),远低于 8-bit 的 ~0.99。
- HQQ 更差(单样本崩到 0.13)。
- 我们的输出 **469 MB**:`MatMulNBitsQuantizer` 只量化 84 个 MatMul,**embedding 表被 skip、保留 fp32**
  (日志里 `embed_tokens/Gather ... skip`)。即比 jina q4f16(~124 MB)大得多,保真度还更低于 8-bit。
  → 4-bit 在保真度与体积两个维度都被 `quantized` 支配。

---

## 实验二:量化算子结构对比(`scripts/inspect_quant.py`)

只读 ONNX graph proto(`load_external_data=False`,不下载 `.onnx_data`),dump 每个
`MatMulNBits`(compute 权重)与 `GatherBlockQuantized`(embedding 表)的 `bits` / `block_size` /
对称性。

**结果**:

| 模型 | MatMul 权重(84 个) | embedding 表 | 对称性 |
|---|---|---|---|
| jina `q4f16` | **4-bit**, block 32 | **4-bit** (GatherBlockQuantized) | **asym** |
| jina `q4` | **4-bit**, block 32 | **4-bit** | **asym** |
| jina `quantized` | 8-bit, block 32 | 8-bit | asym |
| 我们的 RTN / GPTQ | 4-bit, block 32 | **fp32(未量化)** | asym |

**确认**:jina `q4` 的 84 个 matmul 与我们 RTN **逐位相同**(`bits=4, block=32, asym`)。唯一结构差异
是 jina 还把 **token embedding 表压到 4-bit**(`GatherBlockQuantized`),我们保留 fp32。

**推翻**:之前"jina 可能用 symmetric 所以更差"的猜测——两边都是 **asymmetric**(带 zero_points)。

---

## 根因分析

把 bit-width 当唯一自变量,一切自洽:

| 组合 | pooled cosine |
|---|---|
| 4-bit matmul + **4-bit embedding**(jina q4) | ~0.66 |
| 4-bit matmul + **fp32 embedding**(我们 RTN/GPTQ) | 0.79 – 0.81 |
| **8-bit** matmul + **8-bit** embedding(jina quantized) | ~0.99 |

1. **4-bit 是悬崖,matmul 和 embedding 两处都是。** embedding 表对精度极其敏感:8-bit 没问题,
   4-bit 明显掉档(0.79 → 0.66,单变量归因到那 1 个 4-bit 的 `GatherBlockQuantized`)。
2. **last-token pooling 放大误差。** 检索向量是最后一个 token 的 hidden state,不是 mean-pooling,
   逐 token 的量化噪声无法被平均;输入端 4-bit embedding 的噪声经 12 层复合后直接落进那唯一的向量。
3. **nano 模型冗余低**,4-bit(每 block 16 级)相对误差比大模型更伤。
4. **worst-case min 0.178 来自 fp16 计算。** `q4` 与 `q4f16` 结构相同,仅计算精度 fp32 vs fp16 之差;
   平均 cosine 几乎一样,但 fp16 的范围/精度在有 activation outlier 的个别输入上溢出/塌陷,制造单样本崩溃。

一句话:**jina q4/q4f16 = 我们的 4-bit RTN + 额外把 embedding 也压到 4-bit(q4f16 再叠 fp16 计算)**,
在两个悬崖上各踩一脚。它家 `quantized` 是同一拓扑但两处都用 8-bit,躲过了两个悬崖。

---

## 决策(issue #27)

- **发布**:jina 的 8-bit `quant: quantized`(~0.99,~247 MB)作为保真选择。
- **不发布**任何 pipeline 产出的 4-bit target(4-bit 对本模型被支配)。
- **保留能力**:`build_target.sh` 的 `quant: q4gptq` → `optimize_model.py --quant-scheme gptq4`
  校准 4-bit 通路,留给其它能容忍 4-bit 的模型。
- **移除**:`q4` / `q4f16` 两个低保真 release target(它们只在 benchmark 里做对比,保真度不达标)。

---

## 复现命令(容器内)

```bash
# 镜像:docker build -f docker/lambda-build.Dockerfile -t ort-spike .
# 实验一:保真度 spike
docker run --rm -v "$PWD/scripts:/scripts" -v "$PWD/tests/data:/fixtures" -e HF_TOKEN="$HF_TOKEN" \
  ort-spike -c "pip3 install -q onnx-ir && python3 /scripts/spike_gptq_4bit.py \
      --work-dir /tmp/spike --fixture /fixtures/jane-austen_pride-and-prejudice.jsonl"

# 实验二:结构对比(下载 jina q4f16/q4/quantized 的 graph,只读属性)
docker run --rm -v "$PWD/scripts:/scripts" -e HF_TOKEN="$HF_TOKEN" \
  ort-spike -c "python3 /scripts/inspect_quant.py --work-dir /tmp/inspect"
```

## 相关文件

- `scripts/spike_gptq_4bit.py` — 4-bit 保真度 spike(GPTQ/HQQ/RTN vs fp32)。
- `scripts/inspect_quant.py` — 量化算子属性 dump / 对比。
- `scripts/optimize_model.py` — `_step4_gptq_4bit` + `--quant-scheme gptq4`(保留的能力)。
- `scripts/build_target.sh` — `q4gptq` 分支(保留的能力)。
- `builds/release.yaml` — 不含 4-bit target;`quantized` 为保真选择。
