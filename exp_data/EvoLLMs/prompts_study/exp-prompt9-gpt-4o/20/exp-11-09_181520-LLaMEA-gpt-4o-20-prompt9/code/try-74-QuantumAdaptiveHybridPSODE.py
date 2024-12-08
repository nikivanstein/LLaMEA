import numpy as np

class QuantumAdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Slightly increased population for better exploration
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.initial_scale_factor = 0.85  # Slightly adjusted for balance between exploration and exploitation
        self.initial_crossover_rate = 0.8  # Adjusted for better mixing
        self.scale_factor = self.initial_scale_factor
        self.crossover_rate = self.initial_crossover_rate
        self.iterations = self.budget // self.population_size
        self.diversity_threshold = 0.1  # Adjusted for effective diversity control

    def dynamic_parameters(self, evals):
        self.scale_factor = self.initial_scale_factor * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        self.crossover_rate = self.initial_crossover_rate * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))
        self.quantum_weight = 1.5 * np.exp(-evals/self.budget)  # Introduced quantum adaptation

    def compute_diversity(self):
        pairwise_differences = np.sum((self.particles[:, np.newaxis] - self.particles[np.newaxis, :]) ** 2, axis=2)
        return np.mean(np.sqrt(pairwise_differences))

    def quantum_mechanism(self, position):
        # Quantum-inspired position update
        return position + self.quantum_weight * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.iterations):
            self.dynamic_parameters(evaluations)

            for i in range(self.population_size):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)
                inertia = 0.7 - 0.5 * (evaluations / self.budget)  # Fine-tuned inertia adaptation
                cognitive = 1.5 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.5 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            current_diversity = self.compute_diversity()

            for i in range(self.population_size):
                position = np.copy(self.particles[i])
                if current_diversity < self.diversity_threshold:
                    position = self.quantum_mechanism(position)  # Apply quantum update when diversity is low
                score = func(position)
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = position
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = position

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