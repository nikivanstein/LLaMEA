import numpy as np

class DynamicRateEnhancedMultiSwarmEvolutionaryHybridHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def adaptive_mutation(x, F, diversity, fitness_eval, population):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            step_size = F * (fitness_eval + 1e-8) / (diversity + 1e-8)
            swarm_idx = np.random.randint(0, len(population))
            mutant = x + step_size * (population[swarm_idx][np.random.randint(self.budget)] - population[swarm_idx][np.random.randint(self.budget)])
            return np.clip(mutant, -5.0, 5.0)

        def crossover(x, mutant, CR):
            trial = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    trial[i] = mutant[i]
            return trial

        def dynamic_boundary_adjustment(x, lower_bound, upper_bound):
            return np.clip(x, lower_bound, upper_bound)

        num_swarms = 5
        populations = [np.random.uniform(-5.0, 5.0, (self.budget, self.dim)) for _ in range(num_swarms)]
        F = 0.5
        CR = 0.9

        F_adapt = 0.5
        F_lower, F_upper = 0.2, 0.8
        CR_adapt = 0.9
        CR_lower, CR_upper = 0.7, 1.0

        for _ in range(self.budget):
            for swarm_idx in range(num_swarms):
                population = populations[swarm_idx]
                diversity = np.std(population)
                for i in range(self.budget):
                    x = population[i]
                    F = F_adapt + 0.1 * np.random.randn()
                    F = np.clip(F, F_lower, F_upper)
                    CR = CR_adapt + 0.1 * np.random.randn()
                    CR = np.clip(CR, CR_lower, CR_upper)

                    # Updated mutation probability
                    p_mutate = np.clip(0.1 + 0.9 * (1 - np.exp(-fitness_eval)), 0.1, 0.9)
                    if np.random.rand() < p_mutate:
                        mutant = adaptive_mutation(x, F, diversity, cost_function(x), populations)
                    else:
                        mutant = x

                    trial = crossover(x, mutant, CR)
                    if cost_function(trial) < cost_function(x):
                        population[i] = trial
                    else:
                        population[i] = dynamic_boundary_adjustment(x, -5.0, 5.0)

                    if np.random.rand() < 0.1:
                        population[i] = dynamic_boundary_adjustment(x, -5.0, 5.0)

            # Information exchange among swarms
            for i in range(num_swarms):
                exchange_swarm = np.random.choice([x for x in range(num_swarms) if x != i])
                exchange_idx = np.random.randint(self.budget)
                populations[i][exchange_idx] = populations[exchange_swarm][exchange_idx]

        all_populations = np.concatenate(populations)
        best_idx = np.argmin([cost_function(x) for x in all_populations])
        best_solution = all_populations[best_idx]

        return best_solution