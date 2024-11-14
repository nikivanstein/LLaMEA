import numpy as np

class MultiSwarmOptimizer:
    def __init__(self, budget, dim, num_swarm=5):
        self.budget = budget
        self.dim = dim
        self.num_swarm = num_swarm

    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.num_swarm, self.dim)) for _ in range(self.num_swarm)]
        best_positions = [swarm[np.argmin([func(p) for p in swarm])] for swarm in swarms]
        mutation_rates = np.ones(self.num_swarm)

        for _ in range(self.budget):
            for j in range(self.num_swarm):
                for i in range(self.num_swarm):
                    if i != j:
                        candidate_position = np.clip(swarms[j][i] + np.random.normal(0, mutation_rates[j], self.dim), -5.0, 5.0)
                        if func(candidate_position) < func(swarms[j][i]):
                            swarms[j][i] = candidate_position
                            if func(candidate_position) < func(best_positions[j]):
                                best_positions[j] = candidate_position
                                mutation_rates[j] *= 0.99  # Adapt mutation rate based on performance
        return best_positions[np.argmin([func(p) for p in best_positions])]