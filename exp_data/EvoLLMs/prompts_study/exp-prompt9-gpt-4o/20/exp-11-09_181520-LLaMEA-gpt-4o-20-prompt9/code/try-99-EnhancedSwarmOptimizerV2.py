import numpy as np

class EnhancedSwarmOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Adjusted population size for efficiency
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.scale_factor = 0.8  # Fixed scale factor for consistency
        self.crossover_rate = 0.85  # Reduced crossover rate
        self.iterations = self.budget // self.population_size
        self.dynamic_threshold = 0.15  # Dynamic threshold based on diversity measure

    def adaptive_parameters(self, diversity):
        self.scale_factor = 0.8 + 0.2 * (1 - diversity / self.dynamic_threshold)
        self.crossover_rate = 0.85 - 0.2 * (diversity / self.dynamic_threshold)

    def compute_diversity(self):
        pairwise_differences = np.sum((self.particles[:, np.newaxis] - self.particles[np.newaxis, :]) ** 2, axis=2)
        return np.mean(np.sqrt(pairwise_differences))

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            current_diversity = self.compute_diversity()
            self.adaptive_parameters(current_diversity)

            for i in range(self.population_size):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)
                inertia = 0.4 - 0.2 * (evaluations / self.budget)  # Adjusted inertia adaptation
                cognitive = 1.3 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.3 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
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
                    mutant = np.where(np.random.rand(self.dim) < 0.5, mutant, edge_mutation)

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