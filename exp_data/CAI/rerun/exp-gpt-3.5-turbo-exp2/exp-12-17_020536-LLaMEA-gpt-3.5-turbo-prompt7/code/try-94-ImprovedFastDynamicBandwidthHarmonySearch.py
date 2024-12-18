import numpy as np
from joblib import Parallel, delayed

class ImprovedFastDynamicBandwidthHarmonySearch(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for _ in range(2):
                pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
                for i in range(self.dim):
                    if np.random.rand() < pitch_rate:
                        new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            for _ in range(2):
                updated_harmonies = Parallel(n_jobs=-1)(delayed(self._update_harmony_local)(i, func, bandwidth) for i in range(self.dim))
                for i, updated_harmony_local in enumerate(updated_harmonies):
                    if func(updated_harmony_local) < func(self.harmony_memory[-1]):
                        self.harmony_memory[-1] = updated_harmony_local
            updated_harmonies = Parallel(n_jobs=-1)(delayed(self._update_harmony_pitch)(i, func, bandwidth) for i in range(self.dim))
            for updated_harmony_pitch in updated_harmonies:
                if func(updated_harmony_pitch) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = updated_harmony_pitch
            bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]

    def _update_harmony_local(self, i, func, bandwidth):
        new_harmony_local = np.copy(self.harmony_memory[-1])
        new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                 min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
        return new_harmony_local

    def _update_harmony_pitch(self, i, func, bandwidth):
        new_harmony_pitch = np.copy(self.harmony_memory[-1])
        new_harmony_pitch[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                 min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
        return new_harmony_pitch