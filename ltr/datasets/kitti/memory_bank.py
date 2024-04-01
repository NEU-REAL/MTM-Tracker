import numpy as np

class MemoryBank(object):
    def __init__(self, max_length):
        self.max_length = max_length
        self.box_queue = []

    def add(self, new_box):
        # import ipdb;ipdb.set_trace()
        # assert new_box.size == 7
        if self.cur_length() >= self.max_length:
            self.box_queue.pop()
        self.box_queue.insert(0, new_box)

    def return_box(self):
        return np.stack((self.box_queue))

    def cur_length(self):
        return len(self.box_queue)

    def full_length(self):
        return self.max_length

    def reset(self):
        self.box_queue = []
