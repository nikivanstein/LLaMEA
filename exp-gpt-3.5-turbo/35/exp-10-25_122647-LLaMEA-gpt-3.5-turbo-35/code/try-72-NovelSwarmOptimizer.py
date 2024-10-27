import numpy as np

class NovelSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.probability_refinement = 0.35

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        swarm_size = 10
        mutation_rates = np.random.uniform(0, 1, swarm_size)

        for _ in range(self.budget // swarm_size):
            swarm = [np.clip(np.random.normal(best_solution, mutation_rates[i]), -5.0, 5.0) for i in range(swarm_size)]
            fitness_values = [func(individual) for individual in swarm]

            for idx, fitness in enumerate(fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = swarm[idx]

            for i in range(swarm_size):
                ind1, ind2, ind3 = np.random.choice(range(swarm_size), 3, replace=False)
                mutant = best_solution + 0.5 * (swarm[ind1] - swarm[ind2])
                trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, best_solution)
                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            mutation_rates = np.clip(np.random.normal(mutation_rates, 0.1), 0, 1)  # Adapt mutation rates

            if np.random.uniform(0, 1) < self.probability_refinement:
                swarm_size += 1
                mutation_rates = np.append(mutation_rates, np.random.uniform(0, 1))
        
        return best_solution