import numpy as np

class ImprovedHybridDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5 + 0.1 * np.random.rand()  # Dynamic F for diversity
        self.CR = 0.9
        self.omega = 0.5  # Constant inertia for simplicity
        self.phi_p = 1.7  # Adjusted cognitive coefficient
        self.phi_g = 1.7  # Adjusted social coefficient
        self.eval_interval = self.pop_size // 3  # Evaluating less frequently

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evals < self.budget:
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_value = func(trial)
                self.evals += 1
                if trial_value < self.personal_best_values[i]:
                    self.personal_best_positions[i], self.personal_best_values[i] = trial, trial_value
                    if trial_value < self.global_best_value:
                        self.global_best_position, self.global_best_value = trial, trial_value

            if self.evals % self.eval_interval == 0:
                self.evaluate_population(func)

            r_p, r_g = np.random.rand(2, self.pop_size, self.dim)
            self.velocities += (self.phi_p * r_p * (self.personal_best_positions - self.population) +
                                self.phi_g * r_g * (self.global_best_position - self.population))
            self.population = np.clip(self.population + self.omega * self.velocities, self.lower_bound, self.upper_bound)

        return self.global_best_value

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evals >= self.budget:
                break
            value = func(self.population[i])
            self.evals += 1
            if value < self.personal_best_values[i]:
                self.personal_best_positions[i], self.personal_best_values[i] = self.population[i], value
                if value < self.global_best_value:
                    self.global_best_value, self.global_best_position = value, self.population[i]