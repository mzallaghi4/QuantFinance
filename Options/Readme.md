### Greeks Calculation for European Options

Calculation og the Greeks (risk metrics) for European call and put options using the Black-Scholes-Merton model. 
The Greeks (Delta, Gamma, Vega, Theta, and Rho) help traders and analysts understand how option prices react to changes in market variables such as the underlying asset price, volatility, time, and interest rates.


### Black-Scholes Model

The Black-Scholes model is a mathematical model used to price European-style options—options that can only be exercised at expiration.
It also calculates "Greeks" like Delta, Gamma, and etc. 
Delta measures the sensitivity of the option price to small changes in the underlying stock price. 
It provides a closed-form formula to compute the fair price of:
- European call options (right to buy at strike price),
- European put options (right to sell at strike price),
- 
#### Greek Calculations
- Delta (Δ): Price sensitivity to underlying asset 
- Gamma (Γ): Delta sensitivity to underlying price
- Theta (Θ): Time decay sensitivity
- Vega (ν): Volatility sensitivity
- Rho (ρ): Interest rate sensitivity

##### Delta vs Spot Price
* Call Delta: Increases from 0 to 1 as spot price rises
* Put Delta: Decreases from 0 to -1 as spot price rises
* Kink at strike price (K=100) showing maximum sensitivity


 ## Option pricing models

* Binomial Tree Models: Discrete-time model
* Monte Carlo Simulation: Stochastic simulation
* Finite Difference Methods: Numerical PDE solver
* Stochastic Volatility Models
- Heston Model: volatility is vary over time (unlike constant in Black-Scholes).
* Jump Diffusion Models
- Merton’s Jump-Diffusion Model: add random jumps in asset price.


 

