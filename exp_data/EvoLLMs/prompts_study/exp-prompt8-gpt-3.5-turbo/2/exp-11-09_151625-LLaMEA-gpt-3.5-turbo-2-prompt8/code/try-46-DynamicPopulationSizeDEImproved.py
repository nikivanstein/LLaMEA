class DynamicPopulationSizeDEImproved(DifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 10

    def selection(self, population, fitness, target):
        if random.random() < 0.1 and len(population) < 2 * self.population_size:
            new_member = self.init_population(1)
            new_fitness = self.evaluate(new_member, self.func)
            if new_fitness < fitness[target]:
                population.append(new_member)
                fitness.append(new_fitness)
        return super().selection(population, fitness, target)