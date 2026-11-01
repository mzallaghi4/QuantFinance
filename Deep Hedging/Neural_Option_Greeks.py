
import torch
import numpy as np

# Use the same model architecture and preprocessing
# as in Stage 1.
#
# Assumptions:
# model: trained PyTorch pricing network
# scaler: fitted sklearn StandardScaler

model.eval()

def calculate_greeks(spot, strike, maturity, scaler, model):
    """
    Calculate Delta, Gamma and Theta
    using automatic differentiation.

    The network predicts normalized price C/S.
    """

    # Create original input variables
    x_original = torch.tensor(
        [[spot, strike, maturity]],
        dtype=torch.float32
    )

    # Standardize input features
    x_scaled_np = scaler.transform(x_original.numpy())

    x_scaled = torch.tensor(
        x_scaled_np,
        dtype=torch.float32
    )

    # Enable differentiation
    x_scaled.requires_grad_(True)

    # Network output: normalized price C/S
    normalized_price = model(x_scaled)

    # Recover actual option price
    S = x_original[:, 0:1]
    predicted_price = normalized_price * S

    # First derivative with respect to scaled inputs
    gradients = torch.autograd.grad(
        outputs=predicted_price,
        inputs=x_scaled,
        grad_outputs=torch.ones_like(predicted_price),
        create_graph=True
    )[0]

    # Convert derivatives from scaled to original units
    scale = torch.tensor(
        scaler.scale_,
        dtype=torch.float32
    )

    delta = gradients[0, 0] / scale[0]

    gamma_scaled = torch.autograd.grad(
        gradients[0, 0],
        x_scaled,
        create_graph=False
    )[0]

    gamma = gamma_scaled[0, 0] / (scale[0] ** 2)

    theta = -gradients[0, 2] / scale[2]

    return {
        "Option Price": predicted_price.item(),
        "Delta": delta.item(),
        "Gamma": gamma.item(),
        "Theta": theta.item()
    }


# Example
results = calculate_greeks(
    spot=100,
    strike=100,
    maturity=0.5,
    scaler=scaler,
    model=model
)

for name, value in results.items():
    print(f"{name}: {value:.6f}")



def calculate_greeks(spot, strike, maturity, scaler, model):
    model.eval()

    # Original variables remain in the computation graph
    x = torch.tensor(
        [[spot, strike, maturity]],
        dtype=torch.float32,
        requires_grad=True
    )

    # Apply standardization without breaking gradients
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)

    x_scaled = (x - mean) / scale

    # Network predicts C/S
    normalized_price = model(x_scaled)

    # Recover actual option price
    predicted_price = normalized_price * x[:, 0:1]

    # First derivatives
    gradients = torch.autograd.grad(
        predicted_price.sum(),
        x,
        create_graph=True
    )[0]

    delta = gradients[0, 0]

    # Second derivative with respect to spot
    second = torch.autograd.grad(
        gradients[0, 0],
        x,
        retain_graph=True
    )[0]

    gamma = second[0, 0]
    theta = -gradients[0, 2]

    return {
        "Option Price": predicted_price.item(),
        "Delta": delta.item(),
        "Gamma": gamma.item(),
        "Theta": theta.item()
    }
