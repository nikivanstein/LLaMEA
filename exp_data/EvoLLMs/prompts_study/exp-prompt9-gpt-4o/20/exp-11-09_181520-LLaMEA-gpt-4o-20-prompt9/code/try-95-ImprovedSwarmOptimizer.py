import numpy as np

class ImprovedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Slight increase in population for higher diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Expanded velocity range
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.scale_factor = 0.9  # Slightly increased scale factor for more exploration
        self.crossover_rate = 0.8  # Modified crossover rate for diversification
        self.iterations = self.budget // self.population_size
        self.dynamic_threshold = 0.1  # Adjusted threshold for diversity sensitivity

    def adaptive_parameters(self, diversity):
        self.scale_factor = 0.7 + 0.3 * (1 - diversity / self.dynamic_threshold)
        self.crossover_rate = 0.9 - 0.4 * (diversity / self.dynamic_threshold)

    def compute_diversity(self):
        pairwise_differences = np.sum((self.particles[:, np.newaxis] - self.particles[np.newaxis, :]) ** 2, axis=2)
        return np.mean(np.sqrt(pairwise_differences))

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            current_diversity = self.compute_diversity()
            self.adaptive_parameters(current_diversity)

            for i in range(self.population_size):
                r1, r2, r3 = np.random.uniform(size=(3, self.dim))
                inertia = 0.5 - 0.3 * (evaluations / self.budget)  # Adjusted inertia
                cognitive = 1.5 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.5 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                attraction = 0.5 * r3 * (np.mean(self.particles, axis=0) - self.particles[i])  # New swarm attraction strategy
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social + attraction
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                score = func(self.particles[i])
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.particles[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.particles[i]

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.scale_factor * (b - c), self.lower_bound, self.upper_bound)
                    edge_mutation = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    mutant = np.where(np.random.rand(self.dim) < 0.6, mutant, edge_mutation)  # Modified mutation strategy

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