## Deep-Learning Framework for Option Pricing and Hedging

It is essential to examine the fundamental distinction between the classical paradigm (dual martingale representations) and the machine-learning paradigm (primitive functional optimization).
Deep hedging is designed to avoid the need for a unique risk-neutral pricing measure in incomplete markets. 
It directly solves a convex risk-minimization problem using neural-network policies. Rather than eliminating risk-neutral valuation, deep hedging shifts the focus from finding a unique replicating price under a risk-neutral measure to finding a trading strategy that minimizes a specified convex risk measure in an incomplete or frictional market.

#### Deep Neural Network for Option Pricing
Objective: Learn the price of an option ( C(S_t, K, T) ) directly from market data

 Model: Fully Connected Neural Network (MLP)

Inputs:  Spot (S), Strike (K), Maturity (T)

Output:  European Call Option Price (C)



```mermaid
flowchart LR
    subgraph INPUT["Input Layer"]
        S["Spot price S"]
        K["Strike price K"]
        T["Maturity T"]
    end

    subgraph HID["Hidden Layers<br/>(MLP: Dense → BN → LeakyReLU → Dropout)"]
        H["Fully connected MLP blocks"]
    end

    subgraph OUT["Output Layer"]
        Y(["Predicted option price"])
    end

    S & K & T --> H
    H --> Y
```


