import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(60, max(12, int(0.12 * dim)))
        self.de_size = self.swarm_size
        self.particles = np.random.uniform(-5, 5, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full((self.swarm_size,), np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.de_population = np.random.uniform(-5, 5, (self.de_size, dim))
        self.population_scores = np.full((self.de_size,), np.inf)
        self.c1 = 1.6
        self.c2 = 1.4
        self.inertia_weight = 0.6
        self.mutation_factor = 0.85
        self.crossover_probability = 0.85
        self.evaluations = 0
        self.adaptive_factor = 0.025
        self.rank_probability = 0.5
        self.initial_velocity_clamp = 0.5

    def adapt_parameters(self):
        if self.global_best_score < np.inf:
            self.inertia_weight = max(0.3, self.inertia_weight * (1 - self.adaptive_factor))
            self.c1 = min(2.2, self.c1 * (1 + self.adaptive_factor))
            self.c2 = min(2.2, self.c2 * (1 + self.adaptive_factor))
            self.rank_probability = max(0.4, min(0.6, self.rank_probability * (1 + self.adaptive_factor)))
            self.crossover_probability = min(0.9, self.crossover_probability + 0.05 * (1 - self.evaluations / self.budget))

    def stochastic_ranking(self, scores):
        idx = np.argsort(scores)
        for i in range(len(scores) - 1):
            if np.random.uniform() < self.rank_probability:
                if scores[idx[i]] > scores[idx[i + 1]]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]
        return idx

    def optimize_particle_swarm(self, func):
        velocity_clamp = self.initial_velocity_clamp * (1 - self.evaluations / self.budget)
        for i in range(self.swarm_size):
            score = func(self.particles[i])
            self.evaluations += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.global_best_position - self.particles[i]))
            self.velocities[i] = np.clip(self.velocities[i], -velocity_clamp, velocity_clamp)
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], -5, 5)

    def optimize_differential_evolution(self, func):
        ranks = self.stochastic_ranking(self.population_scores)
        for i in ranks:
            candidates = list(range(self.de_size))
            candidates.remove(i)
            a, b, c = self.de_population[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), -5, 5)
            trial = np.where(np.random.rand(self.dim) < self.crossover_probability, mutant, self.de_population[i])
            score = func(trial)
            self.evaluations += 1
            if score < self.population_scores[i]:
                self.population_scores[i] = score
                self.de_population[i] = trial

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.adapt_parameters()
            self.optimize_particle_swarm(func)
            self.optimize_differential_evolution(func)
        
        return self.global_best_position, self.global_best_score