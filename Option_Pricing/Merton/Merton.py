import numpy as np

def merton_simulation_dict(params_dict, n_paths, T, dt):
    """
    Simule le modèle de Merton (jump-diffusion) pour plusieurs actifs en utilisant un dictionnaire de paramètres.

    Args:
        params_dict: Dictionnaire contenant les paramètres pour chaque ticker.
            Format attendu:
            {
                "ticker1": {"S0": ..., "mu": ..., "sigma": ..., "lambda_jump": ..., "mu_J": ..., "sigma_J": ...},
                "ticker2": {"S0": ..., "mu": ..., "sigma": ..., "lambda_jump": ..., "mu_J": ..., "sigma_J": ...},
                ...
            }
        n_paths: Nombre de trajectoires simulées pour chaque actif.
        T: Horizon temporel (en années).
        dt: Pas de temps.

    Returns:
        simulations: Dictionnaire structuré contenant les trajectoires de prix pour chaque ticker.
    """
    simulations = {}
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps, dtype=np.float32)

    for ticker, params in params_dict.items():
        # Récupération des paramètres pour le ticker
        S0 = params["S0"]
        mu = params["mu"]
        sigma = params["sigma"]
        lambda_jump = params["lambda_jump"]
        mu_J = params["mu_J"]
        sigma_J = params["sigma_J"]

        # Initialisation des matrices pour les prix
        S = np.zeros((n_paths, n_steps), dtype=np.float32)

        # Conditions initiales
        S[:, 0] = S0

        # Simuler les mouvements browniens pour la composante diffusion
        Z = np.random.normal(size=(n_paths, n_steps - 1)).astype(np.float32)
        W = Z * np.sqrt(dt)

        # Simuler les sauts (Poisson + tailles des sauts)
        N = np.random.poisson(lambda_jump * dt, size=(n_paths, n_steps - 1)).astype(np.float32)
        jump_sizes = np.random.normal(mu_J, sigma_J, size=(n_paths, n_steps - 1)).astype(np.float32)

        # Boucle temporelle pour simuler les trajectoires
        for i in range(1, n_steps):
            # Composante diffusion
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * W[:, i-1]
            # Composante de saut
            jumps = N[:, i-1] * jump_sizes[:, i-1]
            # Mise à jour du prix
            S[:, i] = S[:, i-1] * np.exp(diffusion + jumps)

        # Construire la structure des trajectoires
        trajectories = []
        for path_id in range(n_paths):
            trajectories.append({
                "path_id": path_id,
                "S": S[path_id, :].tolist()
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
            "mu": 0.05,         # Drift attendu
            "sigma": 0.2,       # Volatilité de la composante diffusion
            "lambda_jump": 0.5, # Intensité des sauts (sauts par an)
            "mu_J": 0.0,        # Moyenne des tailles de saut
            "sigma_J": 0.1      # Écart-type des tailles de saut
        }
    }

    # ⚙️ Simulation
    sim = merton_simulation_dict(params, n_paths=5, T=1, dt=1/252)

    # 📈 Plot des trajectoires de prix pour AAPL
    t = sim["AAPL"]["t"]
    plt.figure(figsize=(10, 5))
    for traj in sim["AAPL"]["trajectories"]:
        plt.plot(t, traj["S"], alpha=0.8)
    plt.title("Simulation du modèle de Merton - AAPL")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix simulé")
    plt.grid(True)
    plt.show()