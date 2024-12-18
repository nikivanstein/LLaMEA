import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5
        self.F = 0.5
        self.NP_min = 5
        self.NP_max = 20

    def __call__(self, func):
        def generate_population(NP):
            return np.random.uniform(-5.0, 5.0, size=(NP, self.dim))

        NP = self.NP_min
        population = generate_population(NP)
        best_solution = population[np.argmin([func(ind) for ind in population])]

        for _ in range(self.budget):
            trial_population = []
            for i in range(NP):
                idxs = [idx for idx in range(NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3])

                if strategy == 0:
                    mutant = population[a] + self.F * (population[b] - population[c])
                elif strategy == 1:
                    mutant = population[a] + self.F * (population[b] - population[c]) + self.F * (population[a] - best_solution)
                elif strategy == 2:
                    mutant = best_solution + self.F * (population[b] - population[c])
                else:
                    mutant = best_solution + self.F * (population[a] - best_solution) + self.F * (population[b] - population[c])

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]

            avg_fitness = np.mean([func(ind) for ind in population])
            NP = min(self.NP_max, max(self.NP_min, int(NP * (1 + (avg_fitness - func(best_solution)) / (avg_fitness + 1e-10))))

        return best_solution