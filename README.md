# Linear Mode Connectivity in Transformer Models

## Overview
With reference to this [paper](https://arxiv.org/abs/1912.05671), this project delves into the exploration of linear mode connectivity within Transformer models, specifically for language modeling tasks on the Penn Treebank dataset. The core objective revolves around understanding the model's stability to Stochastic Gradient Descent (SGD) noise and its implications for linear connectivity.
## How to Use
- The core functions are in `src` directory
    - `src/model.py`
      - define the transformer model in ths class `TransformerModel`
      - `train(model: nn.Module)` and `roll_iter(model: nn.Module)` are used to train one epoch and iterate epochs
      - `analysis(model, start_epoch = None, retrain_init = False)` is used to analyze the instability of two copies of network starting training from `start_epoch`
      - `integrated_analysis` is used for "instability analysis during training" 
    - `src/util.py`
        - include some utility functions for checkpoint saving\loading, and data process.
     
      
1. change directory to src: `src`
2. run the model `python model.py`
3. the log will be saved to `src/runs`, which can be checked on Tensorboard
4. the checkpoint is save to `src/checkpoint`
5. the plots are saved to `graph`

   
## Key Concepts
- **Error Barrier Height Analysis:** A critical measure to assess the stability of the network to SGD noise, focusing on the difference between supreme and expected error barriers.
- **SGD Noise Generation:** The experimentation introduced noise by shuffling the training data batches, aiming to simulate different data ordering scenarios for the optimizer.

## Methodology
1. **Instability Analysis at Initialization:** The process began with the initialization phase, where two copies of a Transformer model, identically initialized but trained on differently shuffled datasets, were analyzed for their response to SGD noise.
2. **Analysis During Training:** The study extended into the training phase, monitoring the model's behavior and stability in response to SGD noise over time.
3. **Stability Thresholds and Metrics:** Instability under 2% was considered stable, with perplexity (ppl) selected as the metric over accuracy for this specific task.

## Results
- Initial analysis showed a notable instability in model behavior, with training and testing perplexity instability at 14.9% and 13.7% respectively.
![PPL vs alpha from initialization](https://github.com/hahacen/linear_mode_connectivity_transformer/assets/103203631/40c01a8f-a731-4ba7-aea5-582330a82b67)
- A significant reduction in instability was observed after the first epoch of training, highlighting the model's evolving response to SGD noise.
![5231710652821_ pic](https://github.com/hahacen/linear_mode_connectivity_transformer/assets/103203631/902dabb1-b9c2-4e66-84d3-81ac20328c69)

  
## Conclusion
The investigation sheds light on the critical aspects of model stability, offering valuable insights into the dynamics of linear mode connectivity in the context of Transformer models and language modeling. This contributes to a deeper understanding of model behavior under the influence of SGD noise, paving the way for further research and optimization strategies.
