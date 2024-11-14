import numpy as np

class EnhancedQuantumInspiredSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocity = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')
        self.alpha = 0.5  # Momentum weight
        self.tau = 0.1  # Quantum tunneling probability

    def __call__(self, func):
        self.global_best = self.quantum_tunneling_initialization(func)

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive momentum update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = r1 * (self.personal_best[i] - self.population[i])
                social_component = r2 * (self.global_best - self.population[i])
                self.velocity[i] = self.alpha * self.velocity[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

                # Quantum tunneling
                if np.random.rand() < self.tau:
                    tunneling_vector = self.global_best + np.random.normal(0, 1, self.dim)
                    tunneling_vector = np.clip(tunneling_vector, self.lower_bound, self.upper_bound)
                    trial_vector = np.copy(tunneling_vector)
                else:
                    trial_vector = np.copy(self.population[i])

                trial_score = func(trial_vector)
                self.func_evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector

            # Adaptive adjustment of alpha and tau
            self.alpha = 0.5 * (1 + np.sin(np.pi * self.func_evaluations / self.budget))
            self.tau = 0.1 * (1 - np.cos(np.pi * self.func_evaluations / self.budget))

        return self.global_best

    def quantum_tunneling_initialization(self, func):
        wave_position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        best_wave_score = float('inf')
        best_wave_position = None

        for pos in wave_position:
            score = func(pos)
            if score < best_wave_score:
                best_wave_score = score
                best_wave_position = pos

        self.func_evaluations += self.population_size
        return best_wave_position