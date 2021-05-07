import numpy as np
from sklearn.neural_network import MLPRegressor

freq_num = 1
output_num = 4 + 1
std_gene = 0.2
std_sense = 1
breeding_age = 5
center = 1
cur_id = 0
hidden_state_size = 0

def chooser(a):
    r = a > center
    return r


class Animal:
    def __init__(self, x, y, s_id=0, pred=False):
        global cur_id
        self.id = cur_id
        self.prev_si = None
        cur_id += 1
        self.obj = []
        self.alive = True
        self.hp = 100
        self.x, self.y = x, y
        self.network = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1, activation='logistic')
        self.network.fit([np.random.random(freq_num * 4 + hidden_state_size + 1)], [np.random.randint(2,size=output_num + hidden_state_size)])
        self.sound_sense = np.zeros(freq_num)
        self.hidden_state = np.random.random(hidden_state_size)
        self.age = 0
        self.sound = None
        self.true_age = 0
        self.species_id = s_id
        self.predator = pred

    def make_decision(self, sound_inputs):
        fsi = [np.concatenate([self.hidden_state, sound_inputs.flatten(), np.array([1])], axis=0)]
        #outs = np.matmul(self.genes, fsi)
        outs = self.network.predict(fsi)[0]
        self.hidden_state = outs[:hidden_state_size]
        outs = outs[hidden_state_size:]
        move_outs = np.exp(outs[:4])
        outs[:4] = 2. * move_outs / np.sum(move_outs)
        assert len(outs.shape) == 1 and outs.shape[0] == output_num
        return outs

    @staticmethod
    def breed(a1, a2, b_loc):
        if a1.age < breeding_age or a2.age < breeding_age or a1.species_id != a2.species_id:
            return False
        bx, by = b_loc
        c = Animal(bx, by, a1.species_id, a1.predator)
        for i in range(len(a1.network.coefs_)):
            selector = np.random.randint(2, size=a1.network.coefs_[i].shape)
            child_weights = a1.network.coefs_[i] * selector + a2.network.coefs_[i] * (1 - selector) + np.random.normal(0, std_gene, size=a1.network.coefs_[i].shape)
            c.network.coefs_[i] = child_weights

        # selector = np.random.randint(2)
        # for i in range(len(a1.network.coefs_)):
        #     child_weights = a1.network.coefs_[i] * selector + a2.network.coefs_[i] * (1 - selector) + np.random.normal(
        #         0, std_gene, size=a1.network.coefs_[i].shape)
        #     c.network.coefs_[i] = child_weights

        c.sound_sense = np.zeros(freq_num)

        return c

