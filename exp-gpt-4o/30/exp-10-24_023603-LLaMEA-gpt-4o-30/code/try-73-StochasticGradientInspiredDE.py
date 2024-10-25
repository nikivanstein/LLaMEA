import numpy as np

class StochasticGradientInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.memory_size = 5
        self.cross_prob = 0.9
        self.F = 0.5
        self.epsilon = 0.01
        self.probability = 0.3

        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.memory = {
            "F": np.full(self.memory_size, self.F),
            "CR": np.full(self.memory_size, self.cross_prob)
        }
        self.memory_index = 0

    def update_memory(self, F, CR):
        self.memory["F"][self.memory_index] = F
        self.memory["CR"][self.memory_index] = CR
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def dynamic_population_adjustment(self, eval_count):
        fraction = eval_count / self.budget
        self.population_size = int(self.initial_population_size * (1 - fraction * 0.2))  # Slightly more reduction

    def multi_elite_selection(self, eval_count):
        elite_size = max(1, int(self.population_size * 0.15))
        elite_indices = np.argsort(self.fitness)[:elite_size]
        return self.population[elite_indices]

    def stochastic_gradient_update(self, individual, gradient, learning_rate=0.01):
        return individual - learning_rate * gradient

    def __call__(self, func):
        eval_count = 0
        best_fitness = np.inf

        while eval_count < self.budget:
            self.dynamic_population_adjustment(eval_count)
            elites = self.multi_elite_selection(eval_count)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = self.population[indices]

                if np.random.rand() < self.probability:
                    F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                    mutant = np.clip(a + F * (b - c + d - e), self.lower_bound, self.upper_bound)
                else:
                    F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                    best_local = np.argmin(self.fitness[indices])
                    a = elites[np.random.randint(len(elites))]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                CR = np.random.choice(self.memory["CR"]) + np.random.rand() * self.epsilon
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if eval_count >= self.budget:
                    break

                if trial_fitness < self.fitness[i]:
                    gradient = (trial - self.population[i]) / (trial_fitness - self.fitness[i] + 1e-8)
                    trial_updated = self.stochastic_gradient_update(trial, gradient)
                    trial_updated = np.clip(trial_updated, self.lower_bound, self.upper_bound)
                    trial_fitness_updated = func(trial_updated)
                    eval_count += 1

                    if trial_fitness_updated < trial_fitness:
                        trial, trial_fitness = trial_updated, trial_fitness_updated

                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        self.update_memory(F, CR)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]