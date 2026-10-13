# Top-1 Mixture-of-Experts GPT

## Abstract

This project implements a Top-1 Mixture-of-Experts (MoE) feed-forward layer within a small GPT-style language model using PyTorch. The implementation explores sparse expert routing, capacity constraints, auxiliary load-balancing loss, and expert utilisation. A dense GPT model is used as a baseline for comparison on the TinyStories dataset. Training behaviour, validation performance, and routing statistics are recorded and visualised.

## Objectives

- Implement a Top-1 Mixture-of-Experts feed-forward layer in a small GPT model.
- Route each token to a single expert using a learned router.
- Apply a capacity factor to limit the number of tokens processed by each expert.
- Implement an auxiliary load-balancing loss.
- Compare the MoE model with a dense GPT baseline.
- Visualise expert utilisation and capacity overflow.

## Model Architecture

The model uses a GPT-style causal language modelling architecture. In the MoE version, the feed-forward network is replaced with a sparse expert layer.

Each MoE layer contains four experts. Each expert uses a feed-forward network with the dimensions:

- Input projection: 128 → 512
- Output projection: 512 → 128

The router maps the 128-dimensional token representation to four expert scores. Top-1 routing selects the expert with the highest router score for each token. The selected expert output is scaled by its router probability.

## Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | TinyStories |
| Vocabulary size | 50,257 |
| Context length | 128 tokens |
| Batch size | 8 |
| Number of MoE layers | 4 |
| Experts per MoE layer | 4 |
| Main capacity factor | 1.25 |
| Auxiliary-loss coefficient | 0.01 |
| Framework | PyTorch |
| Execution environment | Google Colab with CUDA |

The notebook also evaluates a capacity factor of 1.50 to examine the effect of increasing expert capacity.

## Main Results

The following results were recorded in the 500-step controlled comparison.

| Metric | Dense GPT | Top-1 MoE GPT |
|---|---:|---:|
| Training steps | 500 | 500 |
| Training tokens | 512,000 | 512,000 |
| Training time (s) | 15.3057 | 22.6162 |
| Final validation loss | 4.7631 | 4.8068 |
| Validation perplexity | 117.1055 | 122.3395 |

In this experiment, the dense model achieved a slightly lower validation loss and perplexity, while the MoE model took longer to train. These results are specific to the current implementation and training budget.

The equal number of training steps and token positions does not establish an exact FLOP-matched comparison.

## Expert Routing and Capacity

With a batch size of 8 and a context length of 128, each batch contains 1,024 token positions. With four experts and a capacity factor of 1.25, the capacity per expert is:

\[
C = \left\lceil \frac{1024}{4}\times 1.25 \right\rceil = 320
\]

Routing diagnostics over 20 batches showed the following aggregate drop rates:

| MoE layer | Aggregate drop rate |
|---|---:|
| Layer 1 | 1.04% |
| Layer 2 | 15.38% |
| Layer 3 | 4.09% |
| Layer 4 | 1.25% |

Layer 2 experienced the highest token overflow, indicating an imbalance in how tokens were assigned to experts. Increasing the capacity factor can reduce overflow, although it may also increase the amount of computation permitted per batch.

The notebook includes visualisations of expert utilisation, dropped tokens, and capacity-factor comparisons.

## Repository Contents

- `top1_moe_gpt.ipynb` — model implementation, training, evaluation, and visualisations.
- `README.md` — project overview and experimental summary.
- `report/` — intended location for the final IEEE-style research paper.
- `figures/` — intended location for expert-utilisation and capacity-overflow graphs.
- `results/` — intended location for routing statistics and capacity-factor CSV files.

The report, figures, and results folders should be added after the corresponding files have been uploaded.

## How to Run

1. Open `top1_moe_gpt.ipynb` in Google Colab.
2. Select a suitable runtime; a GPU runtime is recommended.
3. Ensure the required Python packages and TinyStories data are available.
4. Run the notebook cells in order.
5. Review the training curves, validation metrics, expert-routing statistics, and capacity-factor experiments.

Runtime and training duration may vary depending on the available hardware.

## Limitations and Future Work

The current experiments use a small model and a limited training budget. Expert utilisation is not perfectly balanced, and token overflow varies across layers. The recorded dense-versus-MoE experiment matches training steps and token positions, but does not establish equal floating-point operations (FLOPs).

Future work could include a verified FLOP-matched comparison, longer training runs, experiments across multiple random seeds, and additional evaluation of routing strategies and capacity factors.

## Reference

W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.

Paper: https://jmlr.org/papers/volume23/21-0998/21-0998.pdf
