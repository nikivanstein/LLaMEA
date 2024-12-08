import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 50
        self.initial_pop_size = 50  # Initial population size (new)
        self.final_pop_size = 30  # Final population size (new)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.f = 0.5  # DE mutation factor
        self.cr = 0.9  # DE crossover rate
        self.population = None
        self.velocity = None
        self.pbest = None
        self.gbest = None
        self.gbest_value = np.inf
        self.evaluations = 0

    def initialize(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.pbest = self.population.copy()
        self.pbest_value = np.full(self.pop_size, np.inf)

    def evaluate(self, func):
        values = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.pop_size
        for i in range(self.pop_size):
            if values[i] < self.pbest_value[i]:
                self.pbest_value[i] = values[i]
                self.pbest[i] = self.population[i].copy()
            if values[i] < self.gbest_value:
                self.gbest_value = values[i]
                self.gbest = self.population[i].copy()

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
        cognitive = self.cognitive_coeff * r1 * (self.pbest - self.population)
        social = self.social_coeff * r2 * (self.gbest - self.population)
        self.velocity = self.inertia_weight * self.velocity + cognitive + social
        self.velocity = self.velocity * np.sin(r2 * np.pi)  # Updated line using chaotic map
        self.population += self.velocity
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

    def differential_evolution(self, func):
        adaptation_rate = 0.1  # Adaptation rate for f and cr
        for i in range(self.pop_size):
            indices = [index for index in range(self.pop_size) if index != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutation_factor = self.f * (1 + np.sin(self.evaluations * np.pi / self.budget))  # Adaptive chaotic mutation
            mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, self.population[i])
            trial_value = func(trial)
            self.evaluations += 1
            if trial_value < self.pbest_value[i]:
                self.pbest_value[i] = trial_value
                self.pbest[i] = trial.copy()
                if trial_value < self.gbest_value:
                    self.gbest_value = trial_value
                    self.gbest = trial.copy()
            if self.evaluations >= self.budget:
                break
        # Dynamic adaptation
        self.f = max(0.1, self.f - adaptation_rate * (self.f - 0.1))
        self.cr = min(0.9, self.cr + adaptation_rate * (0.9 - self.cr))

    def __call__(self, func):
        self.initialize()
        self.evaluate(func)
        while self.evaluations < self.budget:
            # Adaptive inertia weight
            self.inertia_weight = 0.5 + 0.4 * (self.evaluations / self.budget)
            self.update_velocity_position()
            self.evaluate(func)
            self.differential_evolution(func)
            # Dynamic population size adjustment (new)
            self.pop_size = int(self.initial_pop_size - (self.evaluations / self.budget) * (self.initial_pop_size - self.final_pop_size))
            # Population seeding
            if self.evaluations % (self.pop_size * 10) == 0:
                self.population = np.clip(self.population + np.random.uniform(
                    -0.5, 0.5, (self.pop_size, self.dim)) * (self.gbest - self.population), self.lower_bound, self.upper_bound)
        return self.gbest