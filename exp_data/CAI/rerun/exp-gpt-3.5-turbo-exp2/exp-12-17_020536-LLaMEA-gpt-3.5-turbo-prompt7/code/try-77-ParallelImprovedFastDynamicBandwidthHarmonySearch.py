from joblib import Parallel, delayed

class ParallelImprovedFastDynamicBandwidthHarmonySearch(ImprovedFastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        for _ in range(self.budget):
            new_harmonies = Parallel(n_jobs=-1)(delayed(self.generate_new_harmony)(func, pitch_rate, bandwidth) for _ in range(3))
            new_harmonies.sort(key=lambda x: func(x))
            self.harmony_memory[-1] = new_harmonies[0]
            bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]

    def generate_new_harmony(self, func, pitch_rate, bandwidth):
        new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for _ in range(2):
            pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
        for _ in range(2):
            for i in range(self.dim):
                new_harmony_local = np.copy(self.harmony_memory[-1])
                new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
                if func(new_harmony_local) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony_local
        for i in range(self.dim):
            new_harmony_pitch = np.copy(self.harmony_memory[-1])
            new_harmony_pitch[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                     min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
            if func(new_harmony_pitch) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony_pitch
        return new_harmony