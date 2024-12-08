import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f_min = 0.4
        self.f_max = 0.9
        self.cr_min = 0.2
        self.cr_max = 0.95
        self.elite_count = max(2, self.population_size // 10)
        self.entropy_threshold = 0.05
        self.chaos_control = 1.4  # Adjusted chaotic map control parameter for better exploration
        self.dynamic_resize_factor = 0.05  # New dynamic resizing parameter

    def adaptive_parameters(self):
        f = np.random.uniform(self.f_min, self.f_max)
        cr = np.random.uniform(self.cr_min, self.cr_max)
        return f, cr

    def gradient_based_perturbation(self, sol, func):
        grad = np.random.randn(self.dim)  # Simulated gradient using random noise
        step_size = np.linalg.norm(grad) * 0.01  # Scaled step size
        return sol - step_size * grad  # Gradient-based perturbation

    def chaotic_map(self, x):
        return (self.chaos_control * x + 0.8) % 1  # Enhanced chaotic map

    def mutate(self, target_idx, f):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        chaos_factor = self.chaotic_map(np.random.rand())
        mutant = np.clip(a + f * (b - c) * chaos_factor, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, cr):
        crossover = np.random.rand(self.dim) < cr
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, mutant, target)

    def calculate_entropy(self):
        return np.mean(np.std(self.population, axis=0))

    def dynamic_population_resizing(self):
        entropy = self.calculate_entropy()
        if entropy < self.entropy_threshold:
            new_size = int(self.population_size * (1 - self.dynamic_resize_factor))
            self.population = self.population[:new_size]
            self.population_size = new_size
        elif entropy > self.entropy_threshold:
            new_size = int(self.population_size * (1 + self.dynamic_resize_factor))
            additional_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (new_size - self.population_size, self.dim))
            self.population = np.vstack((self.population, additional_individuals))
            self.population_size = new_size

    def __call__(self, func):
        eval_count = 0
        fitness = np.full(self.population_size, np.inf)

        while eval_count < self.budget:
            for i in range(self.population_size):
                f, cr = self.adaptive_parameters()
                target = self.population[i]
                mutant = self.mutate(i, f)
                trial = self.crossover(target, mutant, cr)

                if self.calculate_entropy() < self.entropy_threshold:
                    trial = self.gradient_based_perturbation(trial, func)

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness

                if trial_fitness < func(target):
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            elite_indices = np.argsort(fitness)[:self.elite_count]
            elites = self.population[elite_indices]
            self.population[:self.elite_count] = elites

            self.dynamic_population_resizing()  # Apply dynamic population resizing

        return self.best_solution, self.best_fitness