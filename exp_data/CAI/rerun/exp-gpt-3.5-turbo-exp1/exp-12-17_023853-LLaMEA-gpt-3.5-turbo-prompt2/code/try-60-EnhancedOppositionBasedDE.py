import numpy as np

class EnhancedOppositionBasedDE(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        populations = [generate_population() for _ in range(5)]  # Initialize multiple diverse populations
        best_solutions = [pop[np.argmin([func(ind) for ind in pop])] for pop in populations]

        for _ in range(self.budget):
            trial_populations = []
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3, 4, 5, 6])

                F = self.F * np.exp(-_ / self.budget)
                CR = self.CR * np.exp(-_ / self.budget)

                if strategy == 0:
                    mutant = populations[0][a] + F * (populations[1][b] - populations[2][c])
                elif strategy == 1:
                    mutant = populations[0][a] + F * (populations[1][b] - populations[2][c]) + F * (populations[0][a] - best_solutions[0])
                elif strategy == 2:
                    mutant = best_solutions[0] + F * (populations[1][b] - populations[2][c])
                elif strategy == 3:
                    mutant = best_solutions[0] + F * (populations[0][a] - best_solutions[0]) + F * (populations[1][b] - populations[2][c])
                elif strategy == 4:
                    mutant = populations[0][a] + F * (populations[1][b] - populations[2][c]) + F * (populations[0][a] - best_solutions[0]) + F * (best_solutions[0] - populations[2][c])
                elif strategy == 5:
                    opposite = 2 * best_solutions[0] - populations[0][i]
                    mutant = best_solutions[0] + F * (opposite - populations[1][a])
                else:
                    opposite = 2 * best_solutions[0] - populations[0][i]
                    mutant = best_solutions[0] + F * (opposite - populations[1][a]) + F * (populations[2][b] - best_solutions[0]) + F * (best_solutions[0] - populations[3][c])

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else populations[0][i, j] for j in range(self.dim)])

                if func(trial_ind) < func(populations[0][i]):
                    trial_populations.append(trial_ind)
                else:
                    trial_populations.append(populations[0][i])

            populations[0] = np.array(trial_populations)
            best_solutions[0] = populations[0][np.argmin([func(ind) for ind in populations[0])]

        return best_solutions[0]