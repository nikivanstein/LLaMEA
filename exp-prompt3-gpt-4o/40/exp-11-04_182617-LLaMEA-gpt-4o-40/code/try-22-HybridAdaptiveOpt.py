import numpy as np

class HybridAdaptiveOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Adjusted size for better exploration
        self.initial_temp = 100
        self.final_temp = 0.1
        self.alpha = 0.85  # Slightly altered for adaptive cooling
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.05  # Inspired by SGD

    def differential_evolution(self, population, fitness):
        F = 0.7  # Modified to enhance exploitation
        CR = 0.85
        new_population = np.copy(population)

        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = np.clip(population[a] + F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial_fitness = fitness(trial)
            self.eval_count += 1

            if trial_fitness < fitness(population[i]):
                new_population[i] = trial
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

        return new_population

    def stochastic_gradient_adjustment(self, solution, fitness):
        gradient_steps = np.random.uniform(-1.0, 1.0, self.dim)
        adjusted_solution = np.clip(solution + self.learning_rate * gradient_steps, self.lower_bound, self.upper_bound)
        adjusted_fitness = fitness(adjusted_solution)
        self.eval_count += 1

        if adjusted_fitness < self.best_fitness:
            self.best_fitness = adjusted_fitness
            self.best_solution = adjusted_solution

    def simulated_annealing(self, solution, fitness):
        current_solution = np.copy(solution)
        current_fitness = fitness(current_solution)
        temp = self.initial_temp

        while temp > self.final_temp and self.eval_count < self.budget:
            new_solution = np.clip(current_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            new_fitness = fitness(new_solution)
            self.eval_count += 1

            if new_fitness < current_fitness or np.exp((current_fitness - new_fitness) / temp) > np.random.rand():
                current_solution = new_solution
                current_fitness = new_fitness

                if current_fitness < self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_solution = current_solution

            temp *= self.alpha

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        
        while self.eval_count < self.budget:
            # Apply Differential Evolution
            population = self.differential_evolution(population, func)

            # Apply Stochastic Gradient Adjustment on the best solution found so far
            self.stochastic_gradient_adjustment(self.best_solution, func)

            # Apply Simulated Annealing on the best solution found so far
            self.simulated_annealing(self.best_solution, func)

        return self.best_solution