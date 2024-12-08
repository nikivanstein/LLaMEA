import numpy as np

class EnhancedHybridPSODE_v1:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Slightly increased for improved exploration
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.scale_factor = 0.8  # Adjusted for improved convergence
        self.crossover_rate = 0.9  # Adjusted crossover rate for better exploration
        self.iterations = self.budget // self.population_size
        self.diversity_threshold = 0.1  # Tweaked threshold for enhanced diversity management

    def update_dynamic_params(self, evals):
        self.scale_factor = 0.8 * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        self.crossover_rate = 0.9 * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))

    def compute_diversity(self):
        diffs = np.subtract.outer(self.particles[:, 0], self.particles[:, 0])
        return np.mean(np.abs(diffs))

    def local_search(self, position):
        perturbation = np.random.normal(0, 0.1, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.iterations):
            self.update_dynamic_params(evaluations)

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                inertia = 0.7 - 0.5 * (evaluations / self.budget)  # Adjusted inertia adaptation
                cognitive = 1.6 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.6 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            diversity = self.compute_diversity()

            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.particles[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.particles[i]

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scale_factor * (b - c), self.lower_bound, self.upper_bound)
                
                if diversity < self.diversity_threshold:
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

            # Additional local search for exploiting nearby potential
            if evaluations < self.budget:
                for i in range(self.population_size):
                    local_position = self.local_search(self.particles[i])
                    score = func(local_position)
                    evaluations += 1
                    if score < self.best_personal_scores[i]:
                        self.best_personal_scores[i] = score
                        self.best_personal_positions[i] = local_position
                        if score < self.best_global_score:
                            self.best_global_score = score
                            self.best_global_position = local_position

        return self.best_global_position, self.best_global_score