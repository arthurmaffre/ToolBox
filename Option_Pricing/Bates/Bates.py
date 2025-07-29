import numpy as np

def bates_simulation_dict(params_dict, n_paths, T, dt):
    """
    Simule le modèle de Bates (stochastic volatility + jumps) pour plusieurs actifs en utilisant un dictionnaire de paramètres.

    Args:
        params_dict: Dictionnaire contenant les paramètres pour chaque ticker.
            Format attendu:
            {
                "ticker1": {"S0": ..., "v0": ..., "mu": ..., "kappa": ..., "theta": ..., "sigma": ..., "rho": ..., "lambda_jump": ..., "mu_J": ..., "sigma_J": ...},
                "ticker2": {"S0": ..., "v0": ..., "mu": ..., "kappa": ..., "theta": ..., "sigma": ..., "rho": ..., "lambda_jump": ..., "mu_J": ..., "sigma_J": ...},
                ...
            }
        n_paths: Nombre de trajectoires simulées pour chaque actif.
        T: Horizon temporel (en années).
        dt: Pas de temps.

    Returns:
        simulations: Dictionnaire structuré contenant les trajectoires de prix et de variance pour chaque ticker.
    """
    simulations = {}
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps, dtype=np.float32)

    for ticker, params in params_dict.items():
        # Récupération des paramètres pour le ticker
        S0 = params["S0"]
        v0 = params["v0"]
        mu = params["mu"]
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        rho = params["rho"]
        lambda_jump = params["lambda_jump"]
        mu_J = params["mu_J"]
        sigma_J = params["sigma_J"]

        # Initialisation des matrices pour les prix et la variance
        S = np.zeros((n_paths, n_steps), dtype=np.float32)
        V = np.zeros((n_paths, n_steps), dtype=np.float32)

        # Conditions initiales
        S[:, 0] = S0
        V[:, 0] = v0

        # Simuler les mouvements browniens corrélés
        Z1 = np.random.normal(size=(n_paths, n_steps - 1)).astype(np.float32)
        Z2 = np.random.normal(size=(n_paths, n_steps - 1)).astype(np.float32)
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        # Simuler les sauts (Poisson + tailles des sauts)
        N = np.random.poisson(lambda_jump * dt, size=(n_paths, n_steps - 1)).astype(np.float32)
        jump_sizes = np.random.normal(mu_J, sigma_J, size=(n_paths, n_steps - 1)).astype(np.float32)

        # Pré-calcul de certaines constantes
        max_variance_epsilon = 1e-6  # Évite les valeurs négatives

        # Boucle temporelle pour simuler les trajectoires
        for i in range(1, n_steps):
            # Simulation de la variance v_t (Heston)
            V[:, i] = (
                V[:, i-1]
                + kappa * (theta - V[:, i-1]) * dt
                + sigma * np.sqrt(np.maximum(V[:, i-1], 0)) * W2[:, i-1]
            )
            V[:, i] = np.maximum(V[:, i], max_variance_epsilon)  # Empêche les valeurs négatives

            # Simulation du prix S_t (Heston + jumps)
            diffusion = (mu - 0.5 * V[:, i-1]) * dt + np.sqrt(V[:, i-1]) * W1[:, i-1]
            jumps = N[:, i-1] * jump_sizes[:, i-1]
            S[:, i] = S[:, i-1] * np.exp(diffusion + jumps)

        # Construire la structure des trajectoires
        trajectories = []
        for path_id in range(n_paths):
            trajectories.append({
                "path_id": path_id,
                "S": S[path_id, :].tolist(),
                "V": V[path_id, :].tolist()
            })

        # Ajouter au dictionnaire principal
        simulations[ticker] = {"trajectories": trajectories, "t": t.tolist()}

    return simulations

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 🔧 Paramètres d'exemple pour un seul actif
    params = {
        "AAPL": {
            "S0": 150,          # Prix initial
            "v0": 0.04,         # Variance initiale
            "mu": 0.05,         # Drift attendu
            "kappa": 2.0,       # Vitesse de retour vers la moyenne
            "theta": 0.04,      # Variance de long terme
            "sigma": 0.3,       # Volatilité de la variance
            "rho": -0.7,        # Corrélation entre prix et variance
            "lambda_jump": 0.5, # Intensité des sauts (sauts par an)
            "mu_J": 0.0,        # Moyenne des tailles de saut
            "sigma_J": 0.1      # Écart-type des tailles de saut
        }
    }

    # ⚙️ Simulation
    sim = bates_simulation_dict(params, n_paths=5, T=1, dt=1/252)

    # 📈 Plot des trajectoires de prix pour AAPL
    t = sim["AAPL"]["t"]
    plt.figure(figsize=(10, 5))
    for traj in sim["AAPL"]["trajectories"]:
        plt.plot(t, traj["S"], alpha=0.8)
    plt.title("Simulation du modèle de Bates - AAPL")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix simulé")
    plt.grid(True)
    plt.show()