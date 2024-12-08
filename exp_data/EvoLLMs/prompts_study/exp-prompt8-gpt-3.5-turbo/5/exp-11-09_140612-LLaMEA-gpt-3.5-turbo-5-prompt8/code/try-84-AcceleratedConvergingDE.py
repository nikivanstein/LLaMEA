def diversity_measure(pop):
    return np.mean(np.std(pop, axis=0))

class AcceleratedConvergingDE(AdaptiveConvergingDE):
    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        CR = np.full(pop_size, 0.5)  
        F = np.full(pop_size, 0.5)   
        
        for _ in range(self.budget):
            new_pop = np.copy(pop)
            diversity = diversity_measure(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR[i]:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    if np.random.rand() < 0.1:  
                        CR[i] = np.clip(CR[i] + np.random.normal(0, 0.1), 0, 1)
                        F[i] = np.clip(F[i] + np.random.normal(0, 0.1), 0, 2)
            
            diversity_new = diversity_measure(pop)
            if diversity_new < diversity:  
                if np.random.rand() < 0.1:  
                    pop_size += 1
                    new_member = np.random.uniform(-5.0, 5.0, (1, self.dim))
                    pop = np.append(pop, new_member, axis=0)
                    fitness = np.append(fitness, func(new_member))
                    CR = np.append(CR, np.random.uniform(0, 1))
                    F = np.append(F, np.random.uniform(0, 2))
                elif pop_size > 5:  
                    worst_idx = np.argmax(fitness)
                    pop = np.delete(pop, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    CR = np.delete(CR, worst_idx)
                    F = np.delete(F, worst_idx)
                    pop_size -= 1
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution