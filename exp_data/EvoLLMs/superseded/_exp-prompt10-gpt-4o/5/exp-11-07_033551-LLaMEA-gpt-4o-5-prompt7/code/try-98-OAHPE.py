import numpy as np

class OAHPE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.c1 = 1.4
        self.c2 = 1.6
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _update_velocity(self, velocities, population, personal_best, global_best):
        r1, r2 = np.random.rand(2, self.population_size, self.dim)
        cognitive = self.c1 * r1 * (personal_best - population)
        social = self.c2 * r2 * (global_best - population)
        return velocities * 0.4 + cognitive + social * 0.85

    def _apply_bounds(self, population):
        np.clip(population, self.lower_bound, self.upper_bound, out=population)

    def _differential_mutation(self, population, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[indices]
        self.mutation_factor = 0.7 + 0.3 * (1 - np.linalg.norm(b - c) / (2 * np.sqrt(self.dim)))
        mutant = a + self.mutation_factor * (b - c)
        trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, population[idx])
        return np.clip(trial, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_value = self._evaluate_population(population, func)

        global_best_idx = np.argmin(personal_best_value)
        global_best = personal_best[global_best_idx]
        global_best_value = personal_best_value[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            self.crossover_prob = 0.6 + 0.2 * (1 - evaluations/self.budget)
            velocities = self._update_velocity(velocities, population, personal_best, global_best)
            population += velocities
            self._apply_bounds(population)

            new_population_value = self._evaluate_population(population, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if new_population_value[i] < personal_best_value[i]:
                    personal_best[i] = population[i]
                    personal_best_value[i] = new_population_value[i]

                if personal_best_value[i] < global_best_value:
                    global_best = personal_best[i]
                    global_best_value = personal_best_value[i]

            for i in range(0, self.population_size, 2):
                if evaluations >= self.budget:
                    break
                trial = self._differential_mutation(population, i)
                trial_value = func(trial)
                if trial_value < personal_best_value[i]:
                    evaluations += 1
                    population[i] = trial
                    personal_best[i] = trial
                    personal_best_value[i] = trial_value

                if personal_best_value[i] < global_best_value:
                    global_best = personal_best[i]
                    global_best_value = personal_best_value[i]

            if evaluations > 0.5 * self.budget and self.population_size > 20:
                self.population_size = int(self.population_size * 0.9)

        return global_best