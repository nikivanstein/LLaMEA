import numpy as np
from sklearn.ensemble import RandomForestRegressor

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Reduced to integrate surrogate model
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.initial_scale_factor = 0.8
        self.initial_crossover_rate = 0.85
        self.iterations = self.budget // self.population_size
        self.surrogate_model = RandomForestRegressor(n_estimators=10)
        self.training_data = []
        self.training_scores = []

    def dynamic_parameters(self, evals):
        self.scale_factor = self.initial_scale_factor * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        self.crossover_rate = self.initial_crossover_rate * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.iterations):
            self.dynamic_parameters(evaluations)

            for i in range(self.population_size):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)
                inertia = 0.4 - 0.3 * (evaluations / self.budget)
                cognitive = 1.6 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.6 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.particles[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.particles[i]
                self.training_data.append(self.particles[i])
                self.training_scores.append(score)

            if len(self.training_data) >= 5 * self.dim:  # Train surrogate model after initial data
                self.surrogate_model.fit(self.training_data, self.training_scores)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scale_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]
                if len(self.training_data) >= 5 * self.dim:
                    surrogate_score = self.surrogate_model.predict([trial])[0]
                    if surrogate_score < self.best_personal_scores[i]:
                        score = func(trial)
                        evaluations += 1
                        if score < self.best_personal_scores[i]:
                            self.particles[i] = trial
                            self.best_personal_scores[i] = score
                            self.best_personal_positions[i] = trial
                            if score < self.best_global_score:
                                self.best_global_score = score
                                self.best_global_position = trial

        return self.best_global_position, self.best_global_score