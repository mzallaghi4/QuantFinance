
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import norm

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------
# 1. Market parameters
# -----------------------------------

S0 = 100.0
K = 100.0
T = 1.0
sigma = 0.20
r = 0.0

N_STEPS = 20
N_PATHS = 10000

dt = T / N_STEPS

transaction_cost = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# 2. Simulate GBM price paths
# -----------------------------------

def simulate_paths():

    Z = np.random.normal(
        size=(N_PATHS, N_STEPS)
    )

    S = np.zeros((N_PATHS, N_STEPS + 1))
    S[:, 0] = S0

    for t in range(N_STEPS):

        S[:, t+1] = S[:, t] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z[:, t]
        )

    return S


paths = simulate_paths()

# -----------------------------------
# 3. Black-Scholes Delta
# -----------------------------------

def bs_delta(S, K, tau, sigma, r=0.0):

    if tau <= 0:
        return (S > K).astype(float)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))

    return norm.cdf(d1)


# -----------------------------------
# 4. Neural hedging architecture
# -----------------------------------

class HedgingNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(5, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)


model = HedgingNetwork().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# 5. Prepare training paths
# -----------------------------------

train_paths = torch.tensor(
    paths[:8000],
    dtype=torch.float32,
    device=device
)

# -----------------------------------
# 6. Deep hedging training
# -----------------------------------

EPOCHS = 100

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    S = train_paths

    previous_hedge = torch.zeros(
        S.shape[0],
        1,
        device=device
    )

    cash = torch.zeros(
        S.shape[0],
        device=device
    )

    for t in range(N_STEPS):

        spot = S[:, t]
        next_spot = S[:, t+1]

        tau = T - t * dt

        # Market state
        delta = torch.tensor(
            bs_delta(
                spot.detach().cpu().numpy(),
                K,
                tau,
                sigma,
                r
            ),
            dtype=torch.float32,
            device=device
        )

        gamma = torch.zeros_like(delta)

        time_input = torch.full_like(
            spot,
            tau
        )

        vol_input = torch.full_like(
            spot,
            sigma
        )

        state = torch.stack(
            [
                spot / S0,
                time_input,
                vol_input,
                delta,
                gamma
            ],
            dim=1
        )

        # Neural hedge position
        hedge = model(state).squeeze(1)

        # Transaction costs
        trading_cost = (
            transaction_cost
            * spot
            * torch.abs(hedge - previous_hedge)
        )

        cash = (
            cash * np.exp(r * dt)
            - hedge * (next_spot - spot)
            - trading_cost
        )

        previous_hedge = hedge

    # Terminal short-option payoff
    payoff = torch.relu(
        S[:, -1] - K
    )

    # Hedging error for short option
    pnl = cash - payoff

    loss = torch.mean(pnl ** 2)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}: "
            f"Loss = {loss.item():.6f}"
        )

# -----------------------------------
# 7. Save trained model
# -----------------------------------

torch.save(
    model.state_dict(),
    "deep_hedging_model.pth"
)

print("Deep hedging model saved.")
