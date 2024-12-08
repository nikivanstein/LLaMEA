import numpy as np

class Dynamic_DE_PSO_Optimizer(Enhanced_DE_PSO_Optimizer_Improved):
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9):
        super().__init__(budget, dim, npop, F, CR, w, c1, c2, F_decay, CR_decay, w_decay, c1_decay, c2_decay)
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max

    def __call__(self, func):
        def mutate(pbest, gbest, pop, F, CR):
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                
                # Dynamic mutation strategy
                fitness_values = [func(ind) for ind in pop]
                mean_fitness = np.mean(fitness_values)
                std_fitness = np.std(fitness_values)
                if std_fitness > 0:
                    F = np.clip(np.abs(np.random.normal(F, 0.1)), self.F_min, self.F_max)
                    CR = np.clip(np.abs(np.random.normal(CR, 0.1)), self.CR_min, self.CR_max)

                chaotic_val = chaotic_map(np.random.rand())
                mutant_chaotic = pop[i] + F * chaotic_val
                mutant_chaotic = np.clip(mutant_chaotic, -5.0, 5.0)
                
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant_rand = np.clip(a + F * (b - c), -5.0, 5.0)

                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant_best = np.clip(pop[i] + F * (gbest - pop[i]) + F * (a - b), -5.0, 5.0)
                
                mutant = np.where(np.random.rand(self.dim) < CR, mutant_rand, mutant_best)
                
                if func(mutant) < func(pop[i]):
                    pop[i] = mutant
                if func(mutant) < func(pbest[i]):
                    pbest[i] = mutant
                if func(mutant) < func(gbest):
                    gbest = mutant
            return pop, pbest, gbest