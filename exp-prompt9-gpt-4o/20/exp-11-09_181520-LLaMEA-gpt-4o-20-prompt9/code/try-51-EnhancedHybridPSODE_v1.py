import numpy as np

class EnhancedHybridPSODE_v1:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.initial_scale_factor = 0.8
        self.initial_crossover_rate = 0.9
        self.scale_factor = self.initial_scale_factor
        self.crossover_rate = self.initial_crossover_rate
        self.iterations = self.budget // self.population_size
        self.diversity_threshold = 0.1  # New parameter for diversity management

    def dynamic_parameters(self, evals):
        self.scale_factor = self.initial_scale_factor * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        self.crossover_rate = self.initial_crossover_rate * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))
        self.inertia_weight = 0.9 - 0.5 * (evals / self.budget)  # Adaptive inertia weight
        self.cognitive_coeff = 2.0 - 0.5 * (evals / self.budget) # Adaptive learning rate for personal best

    def compute_diversity(self):
        pairwise_differences = np.sum((self.particles[:, np.newaxis] - self.particles[np.newaxis, :]) ** 2, axis=2)
        return np.mean(np.sqrt(pairwise_differences))

    def local_search(self, position):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        candidate_position = np.clip(position + perturbation, self.lower_bound, self.upper_bound)
        return candidate_position

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.iterations):
            self.dynamic_parameters(evaluations)

            # PSO Part: Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)
                social = 1.5 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = self.inertia_weight * self.velocities[i] + \
                                     self.cognitive_coeff * r1 * (self.best_personal_positions[i] - self.particles[i]) + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            current_diversity = self.compute_diversity()

            # Evaluate particles
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.particles[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.particles[i]

            # DE Part: Mutation and Crossover with edge-based mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scale_factor * (b - c), self.lower_bound, self.upper_bound)
                
                if current_diversity < self.diversity_threshold:
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

            # Local search enhancement
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                local_candidate = self.local_search(self.particles[i])
                score = func(local_candidate)
                evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.particles[i] = local_candidate
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = local_candidate
                    if score < self.best_global_score:
                        self.best_global_score = score
                        self.best_global_position = local_candidate

        return self.best_global_position, self.best_global_score