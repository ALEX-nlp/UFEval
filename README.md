
# FRABench and UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization

This is the official repository for our paper: [UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization](https://arxiv.org/abs/2505.12795).

## Introduction

We propose UFEval, the first unified fine-grained evaluator with task and aspect generalization for four evaluation tasks --- Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. Specifically, (1) We first construct a hierarchical aspect taxonomy encompassing 112 distinct aspects across the aforementioned four tasks. (2) Then, building upon this taxonomy, we create FRABench, a large-scale pairwise evaluation dataset across four tasks comprising 28 sub-tasks and 60.4k pairwise samples with 325k evaluation labels, which is based on our constructed evaluation aspect tree. Our constructed aspect tree includes 112 distinct aspects, categorized into Universal Aspects (UAs) and Task-specific Aspects (TAs). Once the application scenario is specified, human or automated evaluators can simply traverse our aspect tree to select the relevant ones. This design dramatically lowers the cognitive load of defining evaluation standards and promotes consistency across tasks and modalities. (3) Finally, leveraging FRABench, we propose UFEval, a 7B-parameter large multimodal model (LMM), which is the first unified fine-grained evaluator. Our experimental results demonstrate that joint learning across diverse visual tasks and aspects yields significant mutual benefits and generalization capabilities. 

We believe that an ideal evaluator should be applicable to a wider range of scenarios. The comparison between related methods and UFEval is shown in the table below.


| Method           | NLG | IU | IG | ITIG | Text | Image | Aspect | 
|:----------------:|:---:|:--:|:--:|:----:|:----:|:-----:|:------:|
| Auto-J           | ✓   | ✗  | ✗  | ✗    | ✓    | ✗     | 332    | 
| Prometheus 2     | ✓   | ✗  | ✗  | ✗    | ✓    | ✗     | -      | 
| X-Eval           | ✓   | ✗  | ✗  | ✗    | ✓    | ✗     | 27     | 
| Themis           | ✓   | ✗  | ✗  | ✗    | ✓    | ✗     | 50     | 
| ImageReward      | ✗   | ✗  | ✓  | ✗    | ✗    | ✓     | 3      | 
| VisionReward     | ✗   | ✗  | ✓  | ✗    | ✗    | ✓     | 37     | 
| LLaVA-Critic     | ✗   | ✓  | ✗  | ✗    | ✓    | ✗     | –      | 
| **UFEval (ours)** | ✓  | ✓  | ✓  | ✓    | ✓    | ✓     | 112    | 

## FRABench
You can download our FRABench in Huggingface [FRABench](https://huggingface.co/datasets/SPUH/FRABench). We provide the training set for UFEval and the corresponding testing set: FRA-ID, FRA-ID-H, FRA-OOD, FRA-OOD-H, and FRAUAs-OOD.

| Dataset         | Description                                                                                  | Samples  | Split   |
|-----------------|----------------------------------------------------------------------------------------------|----------|---------|
| **Training Set** | Contains 18 sub-tasks, 22 UAs, and 35 TAs for training.  | 255.4k   | Train   |
| **FRA-ID**      | In-domain evaluations with identical tasks and evaluation aspects as the **Training Set**.           | 45.2k    | Test    |
| **FRA-ID-H**    | Manually annotated subset of **FRA-ID** for human alignment evaluation.                      | 6.9k     | Test    |
| **FRA-OOD**     | Out-of-domain evaluations with 10 unseen tasks, 28 UAs, and 27 unseen TAs.                      | 24.4k    | Test    |
| **FRA-OOD-H**   | Manually annotated subset of **FRA-OOD** for human alignment evaluation.                        | 6.0k     | Test    |
| **FRAUAs-OOD**  | Tests generalization to unseen UAs (same tasks as **FRA-ID** but different UAs). | 5.3k     | Test    |

## Usage

### Environment
We use the [llama-factory](https://github.com/hiyouga/LLaMA-Factory) framework for training and [vllm](https://github.com/vllm-project/vllm) library to accelerate model inference.

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install vllm
```

### Model
Our UFEval-7B is now available on Huggingface: [UFEval](https://huggingface.co/SPUH/UFEval)

### Fine-tuning

When using the training set for training, it is necessary to first fill the query, response 1, response 2, and criterion into the template we provided in the sources directory. We recommend using different templates depending on whether the aspects to be evaluated are UAs or TAs for better results.
| Template               | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| **Text_UAs**           | Evaluates text-related UAs: Subtree aspects with **Readability** as root node, **Bias**, **Toxicity**       |
| **Image_UAs**          | Evaluates image-related UAs: Subtree aspects with **Image Quality** as root node, **Bias**, **Toxicity**      |
| **Multi-Image_UAs**    | Evaluates multi-image-related UAs: Subtree aspects with **Image Coherence** as root node, **Bias**, **Toxicity**   |
| **Text-with-Image_UAs**| Evaluates text-with-image-related UAs: Subtree aspects with **Text-Image Relationship** as root node, **Bias**, **Toxicity**   |
| **NLG_TAs**            | Evaluates Natural Language Generation task-related TAs                                          |
| **IU_TAs**             | Evaluates Image Understanding task-related TAs                                                  |
| **IG_TAs**             | Evaluates Image Generation task-related TAs                                                     |
| **ITIG_TAs**           | Evaluates Interleaved Text-and-Image Generation task-related TAs                                |
| **ITIG_TAs_NoInput**   | Evaluates Interleaved Text-and-Image Generation task-related TAs (without input contents)       |

Then organize the data into [OpenAI's format](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html):
```
[
  {
    "messages": [
      {
        "role": "system",
        "content": "System prompt (optional)"
      },
      {
        "role": "user",
        "content": "Human Instruction"
      },
      {
        "role": "assistant",
        "content": "Model Response"
      }
    ],
    "images"[]
  }
]
```

Finally, you can leverage the [distributed training options](https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html) provided by LLaMA-Factory to train your model efficiently.

### Evaluation

We provide two approaches for evaluation:

1. Sequential Label Generation
2. The generate.py script includes evaluation examples for four distinct tasks. Run the following command:
```
python3 ./src/generate.py ./sources/geneval.yaml --task NLG (or IU/IG/ITIG)
```

2. Parallel Label Generation 

Use vllm.py for parallel evaluation with the vllm backend. Organize the data into the same OpenAI format as the training set. Execute:
```
python3 ./src/vllm.py --model_name_or_path MODEL_PATH --dataset DATA_PATH
```

## Citation
```
@article{hong2025frabench,
  title={UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization},
  author={Hong, Shibo and Ying, Jiahao and Liang, Haiyuan and Zhang, Mengdi and Kuang, Jun and Zhang, Jiazheng and Cao, Yixin},
  journal={arXiv preprint arXiv:2505.12795},
  year={2025}
}
```
