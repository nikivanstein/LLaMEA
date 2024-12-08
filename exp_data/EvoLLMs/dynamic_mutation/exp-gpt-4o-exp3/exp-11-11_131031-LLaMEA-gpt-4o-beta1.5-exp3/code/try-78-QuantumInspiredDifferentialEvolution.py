import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        adaptive_factor = 0.1

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            mutation_factor = 0.5 + 0.5 * np.random.rand() * (1 - progress_ratio)
            crossover_rate = 0.7 + 0.3 * np.random.rand() * progress_ratio

            new_population = np.copy(population)

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Quantum-inspired mutation
                quantum_mutant = a + mutation_factor * (np.sin(b) - np.cos(c))
                quantum_mutant = np.clip(quantum_mutant, self.lower_bound, self.upper_bound)

                trial = np.array([
                    quantum_mutant[j] if np.random.rand() < crossover_rate or j == np.random.randint(self.dim) else population[i][j]
                    for j in range(self.dim)
                ])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Adaptive population reduction
            if evaluations < self.budget and evaluations > self.budget * 0.5:
                new_population_size = max(4, int(self.initial_population_size * (1 - adaptive_factor * progress_ratio)))
                sorted_indices = np.argsort(fitness)
                new_population = new_population[sorted_indices[:new_population_size]]
                fitness = fitness[sorted_indices[:new_population_size]]
                population_size = new_population_size

            if evaluations < self.budget * 0.3 and np.random.rand() < 0.05:
                restart_size = int(0.15 * population_size)
                restart_population = np.random.uniform(self.lower_bound, self.upper_bound, (restart_size, self.dim))
                for idx in range(restart_size):
                    restart_fitness = func(restart_population[idx])
                    evaluations += 1
                    if restart_fitness < best_fitness:
                        best_solution = restart_population[idx]
                        best_fitness = restart_fitness
                population[:restart_size] = restart_population

            population = new_population

        return best_solution