import numpy as np

class EAPE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.4  # Slightly reduce cognitive component
        self.c2 = 1.6  # Slightly increase social component
        self.mutation_factor = 0.85  # Increase mutation strength slightly
        self.crossover_prob = 0.75  # Increase crossover probability
        self.decay_rate = 0.99  # Introduce a decay rate for adaptive behavior

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _update_velocity(self, velocities, population, personal_best, global_best):
        r1, r2 = np.random.rand(2, self.population_size, self.dim)
        cognitive = self.c1 * r1 * (personal_best - population)
        social = self.c2 * r2 * (global_best - population)
        velocities = velocities * self.decay_rate  # Apply decay to velocities
        return velocities + cognitive + social

    def _apply_bounds(self, population):
        np.clip(population, self.lower_bound, self.upper_bound, out=population)

    def _differential_mutation(self, population, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[indices]
        mutant = a + self.mutation_factor * (b - c)
        np.clip(mutant, self.lower_bound, self.upper_bound, out=mutant)
        trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, population[idx])
        np.clip(trial, self.lower_bound, self.upper_bound, out=trial)
        return trial

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

            for i in range(self.population_size):
                trial = self._differential_mutation(population, i)
                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_value[i]:
                    population[i] = trial
                    personal_best[i] = trial
                    personal_best_value[i] = trial_value

                if personal_best_value[i] < global_best_value:
                    global_best = personal_best[i]
                    global_best_value = personal_best_value[i]

                if evaluations >= self.budget:
                    break
        
        return global_best