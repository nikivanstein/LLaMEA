def levy_flight(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / np.power(np.abs(v), 1 / beta)
    return 0.01 * step

class EnhancedDynamicMutationOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        # Existing code remains unchanged
        for _ in range(self.budget):
            # Existing code remains unchanged
            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(mutation_factors[i], 0.1), 0.1, 0.9)  # Dynamic mutation factor
                step = levy_flight(self.dim)
                new_sol += step
                new_sol = np.clip(new_sol, -5.0, 5.0)
                # Existing code remains unchanged
        return gbest