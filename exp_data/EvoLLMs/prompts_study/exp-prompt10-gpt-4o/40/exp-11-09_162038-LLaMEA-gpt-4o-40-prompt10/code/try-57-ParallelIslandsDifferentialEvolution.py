import numpy as np

class ParallelIslandsDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.island_count = 5  # Parallel islands for diversity
        self.population_size = max(5, self.budget // (self.island_count * 10))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_range = (0.4, 0.9)  # Range for differential weight
        self.CR_range = (0.6, 0.9)  # Range for crossover probability
        self.local_search_probability = 0.1  # Probability of local search
        self.mutation_strategy = 'rand'  # 'rand' strategy for global exploration

    def __call__(self, func):
        # Initialize multiple island populations
        islands = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                   for _ in range(self.island_count)]
        fitness = [np.array([func(ind) for ind in island]) for island in islands]
        eval_count = self.population_size * self.island_count

        # Track the best solution found across all islands
        best_fitness = np.inf
        best_individual = None

        while eval_count < self.budget:
            for island_index, island in enumerate(islands):
                island_fitness = fitness[island_index]
                best_island_index = np.argmin(island_fitness)
                island_best_individual = island[best_island_index]

                for i in range(self.population_size):
                    # Adaptive F and CR with random selection
                    F = np.random.uniform(*self.F_range)
                    CR = np.random.uniform(*self.CR_range)

                    # 'rand' mutation strategy with elitism
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = island[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < CR, mutant, island[i])

                    # Evaluate trial individual
                    trial_fitness = func(trial)
                    eval_count += 1

                    # Selection and update
                    if trial_fitness < island_fitness[i]:
                        island[i] = trial
                        island_fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_individual = trial
                            best_fitness = trial_fitness

                    if eval_count >= self.budget:
                        break

                # Local search for local refinement
                if np.random.rand() < self.local_search_probability:
                    search_point = island_best_individual + np.random.uniform(-0.05, 0.05, self.dim)
                    search_point = np.clip(search_point, self.lower_bound, self.upper_bound)
                    search_fitness = func(search_point)
                    eval_count += 1
                    if search_fitness < island_fitness[best_island_index]:
                        island[best_island_index] = search_point
                        island_fitness[best_island_index] = search_fitness
                        if search_fitness < best_fitness:
                            best_individual = search_point
                            best_fitness = search_fitness

            # Migrate best individuals between islands for information exchange
            if eval_count + self.island_count < self.budget:
                best_individuals = [island[np.argmin(f)] for island, f in zip(islands, fitness)]
                for i in range(self.island_count):
                    islands[i][0] = best_individuals[i]
                    fitness[i][0] = func(best_individuals[i])
                eval_count += self.island_count

        # Return best found solution
        return best_individual