import numpy as np

class CooperativeCoEvolutionaryOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_candidates = int(budget * 0.2)
        self.initial_step_size = 0.1
        self.num_subpopulations = 2  # Divide the population into subpopulations
        self.subpopulation_size = self.num_candidates // self.num_subpopulations
        self.subpop_solutions = [np.random.uniform(-5.0, 5.0, (self.subpopulation_size, self.dim)) for _ in range(self.num_subpopulations)]
        self.subpop_best_fitness = [float('inf') for _ in range(self.num_subpopulations)]
        self.subpop_stepsizes = [self.initial_step_size for _ in range(self.num_subpopulations)]

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // self.num_candidates):
            for subpop_idx in range(self.num_subpopulations):
                candidate_solutions = self.subpop_solutions[subpop_idx]
                candidate_solutions = [candidate + self.subpop_stepsizes[subpop_idx] * np.random.uniform(-1, 1, self.dim) for candidate in candidate_solutions]
                candidate_fitness = [func(candidate) for candidate in candidate_solutions]

                if min(candidate_fitness) < self.subpop_best_fitness[subpop_idx]:
                    self.subpop_best_fitness[subpop_idx] = min(candidate_fitness)
                    best_in_subpop = candidate_solutions[np.argmin(candidate_fitness)]
                    if self.subpop_best_fitness[subpop_idx] < best_fitness:
                        best_fitness = self.subpop_best_fitness[subpop_idx]
                        best_solution = best_in_subpop

                    self.subpop_solutions[subpop_idx] = candidate_solutions
                    self.subpop_stepsizes[subpop_idx] *= 0.9

            # Exchange information between subpopulations periodically
            if _ % 10 == 0 and _ > 0:
                for subpop_idx in range(self.num_subpopulations):
                    other_subpop_idx = (subpop_idx + 1) % self.num_subpopulations
                    self.subpop_solutions[subpop_idx] = np.concatenate((self.subpop_solutions[subpop_idx][:self.subpopulation_size//2], self.subpop_solutions[other_subpop_idx][self.subpopulation_size//2:]), axis=0)

        return best_solution