# HiMix: Reducing Computational Complexity in Large Vision-Language Models</h1>
[![arXiv](https://img.shields.io/badge/Arxiv-2501.10318-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.10318)
> Xuange Zhang†, Dengjie Li†, Bo Liu, Zenghao Bao, Yao Zhou, Baisong Yang, Zhongying Liu, Yujie Zhong, Zheng Zhao✉️, Tongtong Yuan✉️   

## HiMix Architecture
<p align="center" width="100%">
<a target="_blank"><img src="https://github.com/Xuange923/Xuange923.github.io/blob/main/HiMix/resources/fig-modelMain.png"  style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
  
   Comparison of Vanilla Model and HiMix Architectures. Left: Overall structure of traditional Vanilla. Middle: Overall structure of HiMix. Right: Details of Mixture Attention. 
   
   Hierarchical Vision Injection for Mixture Attention (HiMix) is a method designed to reduce computational overhead while maintaining LVLM performance. After fusing the vision and language features through Mixture Attention, the vision sequence no longer participates in the forward propagation process within the language decoder, thereby substantially decreasing the overall computational load.*anged from smallest to largest, represent the models Qwen2-0.5B, Llama3-1B, TinyLlama-1.1B, Llama3-3B and Vicuna-7B. While maintaining a similar performance to the original models, our HiMix achieves a 10× reduction in computational cost.
</p>



## Contents
- [News](#news)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

## News
- **[2025.1.21]** We release source code of **HiMix**!
- **[2025.1.17]** Our paper: [HiMix: Reducing Computational Complexity in Large Vision-Language Models](https://arxiv.org/abs/2501.10318) is released!


## Environment Setup

1. Clone this repository and navigate to the folder:
    ```bash
    git clone https://github.com/Xuange923/HiMix.git
    cd HiMix
    ```

2. Create a conda environment:
    ```bash
    conda create -n himix python=3.10 -y
    conda activate himix
    pip install --upgrade pip
    ```

3. Install dependencies:
    ```bash
    pip install -e .
    pip install flash-attn --no-build-isolation
    ```

## Data Preparation

### Training Data:
- **Llava-1.5 Version**: Please refer to [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) dataset guide. 
  - Pretrain: `blip_laion_cc_sbu_558k`
  - Finetune: `llava_v1_5_mix665k`
- **Llava-ov Version**: Please refer to [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train) dataset guide. 
  - Pretrain: `LLaVA-OneVision Data mid stage` open-source data.
  - Finetune: `LLaVA-OneVision Data single-image stage` open-source data.

### Evaluation Data:
Please refer to [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) dataset preparation guide for setting up the evaluation data.

## Training
Here’s an example for training LMM using Llama-3.2:

1. Replace data paths in `scripts/train_llama/train_llama.sh` with your local paths.
2. Replace output paths in `scripts/train_llama/train_llama.sh` with your local paths.
3. Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `pretrain_llama.sh` and `finetune_llama.sh`.

Run the training script:
 ```bash
 bash scripts/train_llama/train_llama.sh
 ```

## Evaluation

1. Set your `MODEL_PATH`, `MODEL_NAME`, `EVAL_DIR`, and `CONV_MODE` in `scripts/eval/eval_all.sh`.
2. Run the evaluation:
    ```bash
   bash scripts/eval/eval_all.sh
    ```
   Alternatively, you can run individual dataset evaluation scripts.

3. To test FLOPS, set your `model-path` and `conv_mode` in `scripts/cal_flops.py` and then run the program.

## Results
<p align="center" width="100%">
<a target="_blank"><img src="https://github.com/Xuange923/Xuange923.github.io/blob/main/HiMix/resources/fig-flops.png"  style="width: 50%; min-width: 150px; display: block; margin: auto;"></a>
  
   *Comparison of performance and computational cost of the language decoder between the original and HiMix models. The circles, arranged from smallest to largest, represent the models Qwen2-0.5B, Llama3-1B, TinyLlama-1.1B, Llama3-3B and Vicuna-7B. While maintaining a similar performance to the original models, our HiMix achieves a 10× reduction in computational cost.*
</p>





| Language Decoder | VQA<sup>v2</sup> | GQA | VQA<sup>T</sup> | MM-Vet | POPE | MME | MMMU | Params (B) | FLOPs (G) |
|------------------|------------------|-----|-----------------|--------|------|-----|------|------------|-----------|
| Qwen2-0.5B       | 72.3             | 55.8| 45.2            | 19.5   | 86.6 | 1153.0 | 29.7 | 0.49       | 837       |
| +HiMix           | **72.6**         | **55.9** | 44.2 | **21.1** | 84.9 | 1039.7 | **31.6** | 0.50 | **78 (9%)** |
| TinyLlama-1.1B   | 75.5             | 58.6| 49.6            | 23.5   | 86.3 | 1256.5 | 28.3 | 1.10       | 1750      |
| +HiMix           | **75.5**         | **59.2** | 44.1 | **24.5** | 85.7 | 1179.0 | **29.0** | 1.11 | **161 (9%)** |
| Llama-3.2-1B     | 76.8             | 59.6| 52.7            | 27.8   | 86.7 | 1334.5 | 30.6 | 1.24       | 2040      |
| +HiMix           | 76.2             | **59.6** | 50.5 | **28.4** | 86.6 | 1198.6 | 29.9 | 1.25 | **192 (9%)** |
| Llama-3.2-3B     | 78.6             | 62.5| 55.3            | 31.5   | 86.9 | 1449.3 | 34.8 | 3.21       | 5310      |
| +HiMix           | 78.4             | 61.4| 54.4            | 30.2   | 86.6 | 1261.9 | **35.6** | 3.28 | **525 (9%)** |
| Vicuna-7B<sup>*</sup> | 80.6 | 63.3| 63.7 | 40.0 | 86.6 | 1439.1 | 36.4 | 7.19 | 10800 |
| +HiMix<sup>*</sup> | 80.3 | 62.4 | 61.4 | 37.5 | **87.4** | **1498.9** | **38.3** | 7.47 | **1310 (12%)** |

## Citations
If you find this work helpful, please cite our paper:

```
@misc{zhang2025himixreducingcomputationalcomplexity,
              title={HiMix: Reducing Computational Complexity in Large Vision-Language Models},
              author={Xuange Zhang and Dengjie Li and Bo Liu and Zenghao Bao and Yao Zhou and Baisong Yang and Zhongying Liu and Yujie Zhong and Zheng Zhao and Tongtong Yuan},
              year={2025},
              eprint={2501.10318},
              archivePrefix={arXiv},
              primaryClass={cs.CV},
              url={https://arxiv.org/abs/2501.10318},
```

## Acknowledgments
This project is based on [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory). We sincerely thank them for providing such a clear code structure and training scripts.
