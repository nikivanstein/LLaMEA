import numpy as np

class EnhancedAdaptiveHybridPSODE:
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
        self.opposition_factor = 0.2

    def dynamic_parameters(self, evals):
        self.scale_factor = self.initial_scale_factor * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        self.crossover_rate = self.initial_crossover_rate * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))

    def opposition_based_learning(self, particles):
        oppositions = self.lower_bound + self.upper_bound - particles
        scores = np.array([func(ind) for ind in oppositions])
        return oppositions, scores

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.iterations):
            self.dynamic_parameters(evaluations)
            oppositions, opposition_scores = self.opposition_based_learning(self.particles)
            for i, (opp, opp_score) in enumerate(zip(oppositions, opposition_scores)):
                evaluations += 1
                if opp_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = opp_score
                    self.best_personal_positions[i] = opp
                    if opp_score < self.best_global_score:
                        self.best_global_score = opp_score
                        self.best_global_position = opp
            
            # PSO Part: Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(size=(2, self.dim))
                inertia = 0.7 - 0.3 * (evaluations / self.budget)
                cognitive = 1.5 * r1 * (self.best_personal_positions[i] - self.particles[i])
                social = 1.5 * r2 * (self.best_global_position - self.particles[i]) if self.best_global_position is not None else 0
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            
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

            # DE Part: Mutation and Crossover
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