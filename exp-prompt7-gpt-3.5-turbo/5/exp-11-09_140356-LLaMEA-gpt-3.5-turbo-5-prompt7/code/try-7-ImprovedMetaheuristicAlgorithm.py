class ImprovedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]
            
            # Dynamic mutation strategy
            beta = np.random.uniform(0.5, 1.0, self.dim) * np.exp(-0.01 * _)
            offspring = parent1 + beta * (parent2 - self.population)
            
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring
            
        return self.population[idx[0]]