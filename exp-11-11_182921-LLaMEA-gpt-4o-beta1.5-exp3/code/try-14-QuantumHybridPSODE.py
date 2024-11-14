import numpy as np

class QuantumHybridPSODE:
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
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize best position using quantum wave function
        self.global_best = self.quantum_wave_initialization(func)

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = r1 * (self.personal_best[i] - self.population[i])
                social_component = r2 * (self.global_best - self.population[i])
                self.velocity[i] = 0.5 * self.velocity[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

                # Mutation: DE/rand/1 strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant_vector = x1 + self.F * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                # Selection
                trial_score = func(trial_vector)
                self.func_evaluations += 1

                if trial_score < func(self.population[i]):
                    self.population[i] = trial_vector

                # Update personal and global bests
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector

            # Adaptive adjustment of F and CR
            self.F = 0.5 * (1 + np.sin(np.pi * self.func_evaluations / self.budget))
            self.CR = 0.9 * (1 - np.sin(np.pi * self.func_evaluations / self.budget))

        return self.global_best

    def quantum_wave_initialization(self, func):
        wave_position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        wave_amplitude = (self.upper_bound - self.lower_bound) / 2.0
        wave_phase = np.random.uniform(0, 2 * np.pi, (self.population_size, self.dim))
        wave_function = wave_position + wave_amplitude * np.sin(wave_phase)

        best_wave_score = float('inf')
        best_wave_position = None

        for pos in wave_function:
            score = func(pos)
            if score < best_wave_score:
                best_wave_score = score
                best_wave_position = pos

        self.func_evaluations += self.population_size
        return best_wave_position