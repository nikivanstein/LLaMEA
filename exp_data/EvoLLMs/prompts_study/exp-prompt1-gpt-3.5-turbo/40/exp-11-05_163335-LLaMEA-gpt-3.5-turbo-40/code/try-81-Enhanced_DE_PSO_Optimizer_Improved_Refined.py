class Enhanced_DE_PSO_Optimizer_Improved_Refined(Enhanced_DE_PSO_Optimizer_Improved):
    ...
    def mutate(pbest, gbest, pop, F, CR):
        mutant_pop = []
        for i in range(self.npop):
            idxs = [idx for idx in range(self.npop) if idx != i]
            
            # Opposition-based learning mutation
            opp_pop = 2.0 * gbest - pop[i]
            mutant_opp = np.clip(opp_pop + F * (opp_pop - pop[i]), -5.0, 5.0)
            
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
    ...