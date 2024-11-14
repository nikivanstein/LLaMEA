import numpy as np

class FastConvergingMultiSwarmDynamicMutationAlgorithmSpeed:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        num_swarms = 5
        swarm_size = pop_size // num_swarms
        pop = [np.random.uniform(-5.0, 5.0, (swarm_size, self.dim)) for _ in range(num_swarms)]
        fitness = [np.array([func(ind) for ind in swarm]) for swarm in pop]
        best_solutions = [swarm[np.argmin(f)] for swarm, f in zip(pop, fitness)]
        mutation_rate = 0.2

        for _ in range(int(self.budget - pop_size)):
            diversity = np.mean([np.mean(np.linalg.norm(swarm - np.mean(swarm, axis=0), axis=1)) for swarm in pop])
            mutation_rate = mutation_rate * (1 + 0.01 * np.exp(-diversity))

            offspring = []
            for swarm in pop:
                for i in range(swarm_size):
                    mutant = swarm[i] + mutation_rate * np.random.normal(size=self.dim)
                    offspring.append(mutant)

            offspring_fitness = [np.array([func(ind) for ind in swarm]) for swarm in np.split(np.array(offspring), num_swarms)]
            best_offspring_idx = [np.argmin(f) for f in offspring_fitness]
            for i, idx in enumerate(best_offspring_idx):
                if np.min(offspring_fitness[i]) < np.min(fitness[i]):
                    pop[i][np.argmin(fitness[i])] = offspring[i*swarm_size + idx]
                    fitness[i] = offspring_fitness[i]
                    if offspring_fitness[i][idx] < func(best_solutions[i]):
                        best_solutions[i] = offspring[i*swarm_size + idx]
            
            if np.random.rand() < 0.1:  # Randomly increase population size
                new_pop = np.array([np.random.uniform(-5.0, 5.0, (swarm_size, self.dim)) for _ in range(num_swarms)])
                new_fitness = [np.array([func(ind) for ind in swarm]) for swarm in new_pop]
                replace_swarm = np.argmax([np.min(f) for f in fitness])
                replace_idx = np.argmax(fitness[replace_swarm])
                if np.min(new_fitness[replace_swarm]) < np.min(fitness[replace_swarm]):
                    if np.min(new_fitness[replace_swarm]) < func(best_solutions[replace_swarm]):
                        best_solutions[replace_swarm] = new_pop[replace_swarm][np.argmin(new_fitness[replace_swarm])]
                    pop[replace_swarm][replace_idx] = new_pop[replace_swarm][np.argmin(new_fitness[replace_swarm])]
                    fitness[replace_swarm] = new_fitness[replace_swarm]

        return np.min([func(sol) for sol in best_solutions])