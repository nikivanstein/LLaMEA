class Enhanced_Dynamic_DE_PSO_Optimizer(Dynamic_DE_PSO_Optimizer):
    def __call__(self, func):
        def mutate(pbest, gbest, pop, F, CR):
            mutant_pop = []
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                scale = 1.0 - (1.0 - 0.1) * np.random.rand()  # Adaptive mutation scaling
                mutant = np.clip(a + scale * F * (b - c), -5.0, 5.0)
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
            return pop, pbest, gbest