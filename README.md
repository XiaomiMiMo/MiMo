<div align="center">
  <picture>
    <source srcset="https://github.com/XiaomiMiMo/MiMo/raw/main/figures/Xiaomi_MiMo_darkmode.png?raw=true" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/XiaomiMiMo/MiMo/raw/main/figures/Xiaomi_MiMo.png?raw=true" width="60%" alt="Xiaomi-MiMo" />
  </picture>
</div>

<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Unlocking the Reasoning Potential of Language Model<br/>From Pretraining to Posttraining
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<br/>

<div align="center" style="line-height: 1;">
  |
  <a href="https://huggingface.co/XiaomiMiMo" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://www.modelscope.cn/organization/XiaomiMiMo" target="_blank">🤖️ ModelScope</a>
  &nbsp;|
  <a href="https://arxiv.org/abs/2505.07608" target="_blank">📔 Technical Report</a>
  &nbsp;|
  <br/>
</div>

<br/>

> This code repository is licensed under the [Apache 2.0 License](https://github.com/XiaomiMiMo/MiMo/blob/main/LICENSE).

## I. Introduction

Currently, most successful RL works, including open-source research, rely on relatively large base models, e.g., 32B models, particularly for enhancing code reasoning capabilities. Moreover, it was widely considered that achieving uniform and simultaneous improvements in both mathematical and code capabilities within a small model is challenging. Nonetheless, we believe that the effectiveness of the RL trained reasoning model relies on the inherent reasoning potential of the base model. To fully unlock the reasoning potential of language models, efforts must focus not only on post-training but also on pre-training strategies tailored to reasoning.

In this work, we present MiMo-7B, a series of models trained from scratch and born for reasoning tasks. Our RL experiments from MiMo-7B-Base show that our model possesses extraordinary reasoning potential, even surpassing much larger 32B models. Additionally, we perform RL training on a cold-started SFT model, resulting in MiMo-7B-RL, which demonstrates superior performance on both mathematics and code reasoning tasks, matching the performance of OpenAI o1-mini.

<p align="center">
  <img width="80%" src="https://github.com/XiaomiMiMo/MiMo/raw/main/figures/curve.png?raw=true">
</p>

We open-source MiMo-7B series, including checkpoints of the base model, SFT model, RL model trained from base model, and RL model trained from the SFT model.
We believe this report along with the models will provide valuable insights to develop powerful reasoning LLMs that benefit the larger community.

### 🌟 Highlights

- **Pre-Training: Base Model Born for Reasoning**
  - We optimize the data preprocessing pipeline, enhancing text extraction toolkits and applying multi-dimensional data filtering to increase reasoning pattern density in pre-training data. We also employ multiple strategies to generate massive diverse synthetic reasoning data.
  - We adopt a three-stage data mixture strategy for pre-training. Overall, MiMo-7B-Base is pre-trained on approximately 25 trillion tokens.
  - We incorporate Multiple-Token Prediction as an additional training objective, which enhances model performance and accelerates inference.

- **Post-Training Recipe: Pioneering Reasoning Model**
    - We curate 130K mathematics and code problems as RL training data, which can be verified by rule-based verifiers. Each problem undergoes careful cleaning and difficulty assessment to ensure quality. We employ only rule-based accuracy rewards to avoid potential reward hacking.
    - To mitigate the sparse reward issue for challenging code problems, we introduce a test difficulty driven code reward. By assigning fine-grained scores for test cases with varying difficulty levels, the policy can be more effectively optimized via dense reward signal.
    - We implement a data re-sampling strategy for easy problems to enhance rollout sampling efficiency and stabilize policy updates, particularly in the later phases of RL training.

- **RL Infrastructure**
    - We develop a Seamless Rollout Engine to accelerate RL training and validation. Our design integrates continuous rollout, asynchronous reward computation, and early termination to minimize GPU idle time, achieving $2.29\times$ faster training and $1.96\times$ faster validation.
    - We support MTP in vLLM and enhance the robustness of the inference engine in the RL system.

## II. Model Details

The MTP layers of MiMo-7B is tuned during pretraining and SFT and freezed during RL. With one MTP layer for speculative decoding, the acceptance rate is about 90%.

<p align="center">
  <img width="80%" src="https://github.com/XiaomiMiMo/MiMo/raw/main/figures/architecture.png?raw=true">
</p>

> Models are available at [https://huggingface.co/XiaomiMiMo](https://huggingface.co/XiaomiMiMo) and [https://www.modelscope.cn/organization/XiaomiMiMo](https://www.modelscope.cn/organization/XiaomiMiMo)

|    **Model**    |                                **Description**                                |                            **Download (HuggingFace)**                             |                                  **Download (ModelScope)**                                  |
| :-------------: | :---------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
|  MiMo-7B-Base   |               Base model with extraordinary reasoning potential               |    [🤗 XiaomiMiMo/MiMo-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base)    |    [🤖️ XiaomiMiMo/MiMo-7B-Base](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-Base)    |
| MiMo-7B-RL-Zero |                       RL model trained from base model                        | [🤗 XiaomiMiMo/MiMo-7B-RL-Zero](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL-Zero) | [🤖️ XiaomiMiMo/MiMo-7B-RL-Zero](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-RL-Zero) |
|   MiMo-7B-SFT   |                       SFT model trained from base model                       |     [🤗 XiaomiMiMo/MiMo-7B-SFT](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT)     |     [🤖️ XiaomiMiMo/MiMo-7B-SFT](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-SFT)     |
|   MiMo-7B-RL    | RL model trained from SFT model, superior performance matching OpenAI o1-mini |      [🤗 XiaomiMiMo/MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)      |      [🤖️ XiaomiMiMo/MiMo-7B-RL](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-RL)      |

## III. Evaluation Results

| Benchmark                     | GPT-4o-0513 | Claude-3.5-Sonnet-1022 | OpenAI o1-mini | QwQ-32B-Preview | R1-Distill-Qwen-14B | R1-Distill-Qwen-7B | MiMo-7B-RL |
| ----------------------------- | :---------: | :--------------------: | :------------: | :-------------: | :-----------------: | :----------------: | :--------: |
| **General**                   |             |                        |                |                 |                     |                    |            |
| GPQA Diamond<br/>(Pass@1)     |    49.9     |          65.0          |      60.0      |      54.5       |        59.1         |        49.1        |    54.4    |
| SuperGPQA<br/>(Pass@1)        |    42.4     |          48.2          |      45.2      |      43.6       |        40.6         |        28.9        |    40.5    |
| DROP<br/>(3-shot F1)          |    83.7     |          88.3          |      83.9      |      71.2       |        85.5         |        77.0        |    78.7    |
| MMLU-Pro<br/>(EM)             |    72.6     |          78.0          |      80.3      |      52.0       |        68.8         |        53.5        |    58.6    |
| IF-Eval<br/>(Prompt Strict)   |    84.3     |          86.5          |      84.8      |      40.4       |        78.3         |        60.5        |    61.0    |
| **Mathematics**               |             |                        |                |                 |                     |                    |            |
| MATH-500<br/>(Pass@1)         |    74.6     |          78.3          |      90.0      |      90.6       |        93.9         |        92.8        |    95.8    |
| AIME 2024<br/>(Pass@1)        |     9.3     |          16.0          |      63.6      |      50.0       |        69.7         |        55.5        |    68.2    |
| AIME 2025<br/>(Pass@1)        |    11.6     |          7.4           |      50.7      |      32.4       |        48.2         |        38.8        |    55.4    |
| **Code**                      |             |                        |                |                 |                     |                    |            |
| LiveCodeBench v5<br/>(Pass@1) |    32.9     |          38.9          |      53.8      |      41.9       |        53.1         |        37.6        |    57.8    |
| LiveCodeBench v6<br/>(Pass@1) |    30.9     |          37.2          |      46.8      |      39.1       |        31.9         |        23.9        |    49.3    |

MiMo-7B series

| Benchmark                     | MiMo-7B-Base | MiMo-7B-RL-Zero | MiMo-7B-SFT | MiMo-7B-RL |
| ----------------------------- | :----------: | :-------------: | :---------: | :--------: |
| **Mathematics**               |              |                 |             |            |
| MATH500<br/>(Pass@1)          |     37.4     |      93.6       |    93.0     |    95.8    |
| AIME 2024<br/>(Pass@1)        |     32.9     |      56.4       |    58.7     |    68.2    |
| AIME 2025<br/>(Pass@1)        |     24.3     |      46.3       |    44.3     |    55.4    |
| **Code**                      |              |                 |             |            |
| LiveCodeBench v5<br/>(Pass@1) |     32.9     |      49.1       |    52.3     |    57.8    |
| LiveCodeBench v6<br/>(Pass@1) |     29.1     |      42.9       |    45.5     |    49.3    |

> [!IMPORTANT]
> The evaluations are conducted with `temperature=0.6`.
> 
> AIME24 and AIME25 are with averaged score of 32 repetitions. LiveCodeBench v5 (20240801-20250201), LiveCodeBench v6 (20250201-20250501), GPQA-Diamond and IF-Eval are with averaged score of 8 repetitions. MATH500 and SuperGPQA are with a single run.

## IV. Deployment

### SGLang Inference

Thanks to the [MiMo model support](https://github.com/sgl-project/sglang/pull/5921) and [MTP](https://github.com/sgl-project/sglang/pull/6059) from the SGLang team, we supported MiMo in SGLang mainstream.

Example Script

```bash
# Install the latest SGlang from main branch
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python"

# Launch SGLang Server
python3 -m sglang.launch_server --model-path XiaomiMiMo/MiMo-7B-RL --host 0.0.0.0 --trust-remote-code

# Launch MTP Server
python3 -m sglang.launch_server --model-path XiaomiMiMo/MiMo-7B-RL --trust-remote-code \
--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 \
--speculative-num-draft-tokens 2  --mem-fraction 0.5
```

Detailed usage can be found in [SGLang documents](https://docs.sglang.ai/backend/send_request.html).

### vLLM inference

1. [Recommended] We officially support inference with MiMo-MTP using [our fork of vLLM](https://github.com/XiaomiMiMo/vllm/tree/feat_mimo_mtp_stable_073).

Example script

```py
from vllm import LLM, SamplingParams

model_path = "/path/to/MiMo"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    num_speculative_tokens=1,
    disable_log_stats=False
)
sampling_params = SamplingParams(temperature=0.6)

conversation = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("=" * 80)
```

2. Or, you can register a vLLM loader for MiMo without loading MTP parameters.

You can copy the [`registry/register_mimo_in_vllm.py`](https://github.com/XiaomiMiMo/MiMo/blob/main/registry/register_mimo_in_vllm.py) to your directory and import it with

```py
import register_mimo_in_vllm

from vllm import LLM, SamplingParams

model_path = "/path/to/MiMo"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    # num_speculative_tokens=1,
    disable_log_stats=False
)
sampling_params = SamplingParams(temperature=0.6)
```

### HuggingFace inference

Example script

```py
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model_id = "XiaomiMiMo/MiMo-7B-RL"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(["Today is"], return_tensors='pt')
output = model.generate(**inputs, max_new_tokens = 100)
print(tokenizer.decode(output.tolist()[0]))
```

### Recommended environment and prompts

- We recommend using [our fork of vLLM](https://github.com/XiaomiMiMo/vllm/tree/feat_mimo_mtp_stable_073) which is developed based on vLLM 0.7.3.
- We recommend using empty system prompt.

> We haven't verified MiMo with other inference engines and welcome contributions based on the model definition in the Huggingface repo 💻.

## V. Citation

```bibtex
@misc{coreteam2025mimounlockingreasoningpotential,
      title={MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining}, 
      author={{Xiaomi LLM-Core Team}},
      year={2025},
      eprint={2505.07608},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07608}, 
}
```


## VI. Contact

Please contact us at [mimo@xiaomi.com](mailto:mimo@xiaomi.com) or open an issue if you have any questions.
