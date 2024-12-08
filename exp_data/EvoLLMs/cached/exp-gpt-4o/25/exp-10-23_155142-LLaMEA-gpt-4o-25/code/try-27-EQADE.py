import numpy as np

class EQADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 5 * dim  # Adjusted initial population size
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory = []
        self.best_solution = None

    def quantum_initialize(self):
        # Quantum-inspired initialization
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def __call__(self, func):
        # Initialize population
        population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive mutation strategy with elitist adjustment
            if len(self.memory) > 0:
                self.F = np.mean([entry[0] for entry in self.memory])
                self.Cr = np.mean([entry[1] for entry in self.memory])
                if eval_count % (self.pop_size * 2) == 0:  # Adjust population size dynamically
                    self.pop_size = max(5, int(self.pop_size * 0.9))
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]

            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation with best individual consideration
                if self.best_solution is not None:
                    x_best = self.best_solution
                else:
                    x_best = population[np.argmin(fitness)]
                
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x_best + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((self.F, self.Cr))
                    if len(self.memory) > 100:  # Maintain limited memory size
                        self.memory = sorted(self.memory, key=lambda x: x[0])[:50]  # Elitist memory culling

                    # Update best solution
                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]