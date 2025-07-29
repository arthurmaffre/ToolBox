# 📘 Stochastic Financial Models Simulation Library

This repository provides Python implementations for simulating various stochastic models used in financial mathematics, particularly for asset price dynamics. The models are designed to generate Monte Carlo trajectories for prices (and variances where applicable) over a specified time horizon. Each model is implemented in a separate Python file, following a consistent interface for ease of use.

The library includes:
- **Black-Scholes Model**: Basic geometric Brownian motion (constant volatility, no jumps).
- **Merton Jump-Diffusion Model**: Extends Black-Scholes with jumps to capture sudden price changes.
- **Heston Model**: Extends Black-Scholes with stochastic volatility to model volatility clustering.
- **Bates Model**: Combines Heston (stochastic volatility) and Merton (jumps) for more realistic dynamics.

These models are useful for options pricing, risk management, and portfolio simulations. Simulations use NumPy for efficiency and can handle multiple assets via a parameter dictionary.

## 📄 Model Hierarchy and Progression
The models follow a logical progression of increasing complexity, building on the foundational Black-Scholes model to address real-world market behaviors like jumps and time-varying volatility:

1. **Black-Scholes (BS) Model**:
   - Assumes constant volatility and no jumps.
   - Ideal for simple scenarios but limited in capturing market realities.

2. **Merton Jump-Diffusion Model**:
   - Adds jumps to Black-Scholes via a compound Poisson process.
   - Better for assets with occasional large price movements (e.g., due to news events).

3. **Heston Model**:
   - Introduces stochastic volatility using a mean-reverting CIR process.
   - Captures volatility smiles and clustering observed in markets.

4. **Bates Model**:
   - Merges Heston and Merton: stochastic volatility plus jumps.
   - Suitable for advanced options pricing and fitting implied volatility surfaces.

For models beyond Bates (e.g., Lévy processes or rough volatility), consider extending this library or using specialized packages like `stochastic` or `quantlib`.

## ⚙️ Common Interface
All models share a consistent API to simulate trajectories for one or more assets. The main function for each model is named `{model}_simulation_dict` (e.g., `heston_simulation_dict`), taking the following arguments:

- `params_dict`: A dictionary of parameters for each asset (ticker as key). Example for Heston:
  ```python
  {
      "AAPL": {
          "S0": 150,      # Initial price
          "v0": 0.04,     # Initial variance
          "mu": 0.05,     # Drift
          "kappa": 2.0,   # Mean reversion speed
          "theta": 0.04,  # Long-term variance
          "sigma": 0.3,   # Volatility of variance
          "rho": -0.7     # Correlation between price and variance
      }
  }
  ```
  Parameters vary by model (see individual READMEs or docstrings for details).

- `n_paths`: Number of simulated paths (trajectories) per asset (e.g., 1000 for Monte Carlo).

- `T`: Time horizon in years (e.g., 1 for one year).

- `dt`: Time step (e.g., `1/252` for daily steps assuming 252 trading days).

### Return Value
A dictionary with simulations for each asset:
```python
{
    "AAPL": {
        "trajectories": [
            {"path_id": 0, "S": [150.0, 150.5, ...], "V": [0.04, 0.041, ...]},  # "V" only for models with variance
            ...
        ],
        "t": [0.0, 0.003968, ..., 1.0]  # Time grid
    }
}
```
- `S`: List of price values along the path.
- `V`: List of variance values (for Heston and Bates only).
- Use this output for analysis, plotting, or pricing derivatives.

## 🚀 Usage Example
Each model file includes a self-contained example in the `__main__` block for quick testing and plotting with Matplotlib. For instance, in `Heston.py`:
```python
if __name__ == "__main__":
    # Define params, run simulation, and plot trajectories
```
To use across models:
```python
from Heston import heston_simulation_dict
from Merton import merton_simulation_dict
# ... import others

params = {"AAPL": {...}}  # Define parameters
sim_heston = heston_simulation_dict(params, n_paths=100, T=1, dt=1/252)
# Analyze or plot sim_heston["AAPL"]["trajectories"]
```

### Multi-Asset Simulation
Pass multiple tickers in `params_dict` to simulate several assets in one call. Each asset's parameters can differ.

### Dependencies
- NumPy (for simulations and random number generation).
- Matplotlib (for example plots; optional for core usage).

Install via: `pip install numpy matplotlib`

## 📂 Project Structure
- `BlackScholes.py`: Implementation of the Black-Scholes model.
- `Heston.py`: Implementation of the Heston model.
- `Merton.py`: Implementation of the Merton model.
- `Bates.py`: Implementation of the Bates model.
- `README.md`: This file.
- `img/`: Placeholder for example plots (e.g., `Heston_Fig.png`).

## 📊 Example Output
Running the example in any file will generate plots like:
- Trajectories showing price paths over time, with potential jumps (Merton/Bates) or varying volatility (Heston/Bates).

## 🛠️ Extending the Library
- Add new models by following the same function signature and structure.
- For performance, consider vectorizing further or using GPU acceleration (e.g., with CuPy).
- Calibration: These are simulation-only; for parameter estimation, integrate with optimization libraries like SciPy.

## ⚠️ Notes and Limitations
- Simulations use Euler discretization, which may introduce bias for small `dt`; refine `dt` for accuracy.
- No risk-neutral measure adjustments; adapt `mu` for pricing if needed.
- Random seeds are not set by default; use `np.random.seed()` for reproducibility.
- These are educational implementations; for production, validate against established libraries.

Made with ⚡ for faster financial simulations. Contributions welcome!