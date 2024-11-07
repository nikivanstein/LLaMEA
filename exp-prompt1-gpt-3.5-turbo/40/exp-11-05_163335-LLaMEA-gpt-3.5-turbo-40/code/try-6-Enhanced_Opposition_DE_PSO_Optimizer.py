import numpy as np

class Enhanced_Opposition_DE_PSO_Optimizer(Dynamic_DE_PSO_Optimizer):
    def mutate(pbest, gbest, pop, F, CR):
        mutant_pop = []
        for i in range(self.npop):
            idxs = [idx for idx in range(self.npop) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), -5.0, 5.0)
            if np.random.rand() < CR:
                mutant = np.clip(mutant, -5.0, 5.0)
            else:
                mutant = pop[i]
            if func(mutant) < func(pop[i]):
                pop[i] = mutant
            if func(mutant) < func(pbest[i]):
                pbest[i] = mutant
            if func(mutant) < func(gbest):
                gbest = mutant
            # Opposite direction mutation
            opposite_mutant = np.clip(a - F * (b - c), -5.0, 5.0)
            if np.random.rand() < CR:
                opposite_mutant = np.clip(opposite_mutant, -5.0, 5.0)
            else:
                opposite_mutant = pop[i]
            if func(opposite_mutant) < func(pop[i]):
                pop[i] = opposite_mutant
            if func(opposite_mutant) < func(pbest[i]):
                pbest[i] = opposite_mutant
            if func(opposite_mutant) < func(gbest):
                gbest = opposite_mutant
        return pop, pbest, gbest