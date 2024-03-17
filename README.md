# Linear Mode Connectivity in Transformer Models

## Overview
This project delves into the exploration of linear mode connectivity within Transformer models, specifically for language modeling tasks on the Penn Treebank dataset. The core objective revolves around understanding the model's stability to Stochastic Gradient Descent (SGD) noise and its implications for linear connectivity.

## Key Concepts
- **Error Barrier Height Analysis:** A critical measure to assess the stability of the network to SGD noise, focusing on the difference between supreme and expected error barriers.
- **SGD Noise Generation:** The experimentation introduced noise by shuffling the training data batches, aiming to simulate different data ordering scenarios for the optimizer.

## Methodology
1. **Instability Analysis at Initialization:** The process began with the initialization phase, where two copies of a Transformer model, identically initialized but trained on differently shuffled datasets, were analyzed for their response to SGD noise.
2. **Analysis During Training:** The study extended into the training phase, monitoring the model's behavior and stability in response to SGD noise over time.
3. **Stability Thresholds and Metrics:** Instability under 2% was considered stable, with perplexity (ppl) selected as the metric over accuracy for this specific task.

## Results
- Initial analysis showed a notable instability in model behavior, with training and testing perplexity instability at 14.9% and 13.7% respectively.
- A significant reduction in instability was observed after the first epoch of training, highlighting the model's evolving response to SGD noise.

## Use
- The core functions are in scr/model.py, with some utility functions in scr/util.py. I save the model checkpoint in src/checkpoint/, the results are plotted in src/graph.
  
## Conclusion
The investigation sheds light on the critical aspects of model stability, offering valuable insights into the dynamics of linear mode connectivity in the context of Transformer models and language modeling. This contributes to a deeper understanding of model behavior under the influence of SGD noise, paving the way for further research and optimization strategies.
