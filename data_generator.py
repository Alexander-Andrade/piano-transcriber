import numpy as np
import keras
import yaml
import math
import matplotlib.pyplot as plt
from shared import *


class DataGenerator(keras.utils.Sequence):

    def __init__(self, info, n_frames, batch_size, with_labels=True):
        self.info = info
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.with_labels = with_labels
        self.__remove_slices_from_info()
        self.length = self.__calc_len()
        self.__init_iter()

    def __remove_slices_from_info(self):
        for inf in self.info:
            i = 0
            while i < len(inf['slices']):
                el = inf['slices'][i]
                if len(el) == 3:
                    inf['slices'].pop(i)
                    slices_from_range = self.__range_to_slices(el)
                    inf['slices'][i:1] = slices_from_range
                    i += len(slices_from_range)
                else:
                    i += 1

    def __range_to_slices(self, r):
        n_slices = math.ceil((r[1] - r[0]) / self.n_frames)
        slices = []
        for i in range(n_slices):
            start = r[0] + i*self.n_frames
            # [start, shift]
            slices.append([start, r[2]])
        return slices

    def __calc_len(self):
        length = 0
        for file_inf in self.info:
            length += len(file_inf['slices'])
        return length

    def __iter__(self):
        self.__init_iter()
        return self

    def __init_iter(self):
        self.cur = {
            'file_ind': 0,
            'spectrum': None,
            'labels': None,
            'slices_ind': 0
        }

    def __curr_slices(self):
        return self.info[self.cur['file_ind']]['slices']

    def __cur_slice(self):
        return self.__curr_slices()[self.cur['slices_ind']]

    def __next__(self):
        if self.cur['file_ind'] < len(self.info):
            if self.cur['slices_ind'] == 0:
                self.__load_current_file(self.info[self.cur['file_ind']]['filename'])

            cur_slice = self.__cur_slice()
            result = self.__data_chunk(cur_slice)

            if self.cur['slices_ind'] < len(self.__curr_slices()) - 1:
                self.cur['slices_ind'] += 1
            else:
                self.cur['file_ind'] += 1
                self.cur['slices_ind'] = 0

            return result
        else:
            raise StopIteration

    def __data_chunk(self, slice_info):
        cur_slice_pos, shift = slice_info
        spectrum_slice = self.cur['spectrum'][cur_slice_pos:cur_slice_pos + self.n_frames]
        spectrum_slice = spectrum_slice.reshape(self.batch_size, self.n_frames, spectrum_slice.shape[1])

        if self.with_labels:
            labels_slice = shifted_slice(self.cur['labels'], cur_slice_pos + shift, cur_slice_pos + self.n_frames + shift)
            labels_slice = labels_slice.reshape(self.batch_size, self.n_frames, N_NOTES)
            return spectrum_slice, labels_slice

        return spectrum_slice

    def __load_current_file(self, filename):
        self.cur['spectrum'] = np.load("datasets/features_{0}.npy".format(filename))
        if self.with_labels:
            self.cur['labels'] = np.load("datasets/labels_{0}.npy".format(filename))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        slices_rel = 0
        for file_ind, file in enumerate(self.info):
            file_slices_len = len(file['slices'])
            if slices_rel <= idx < slices_rel + file_slices_len:
                if file_ind != self.cur['file_ind'] or self.cur['spectrum'] is None:
                    self.__load_current_file(file['filename'])
                    self.cur['file_ind'] = file_ind
                slice_info = file['slices'][idx - slices_rel]
                return self.__data_chunk(slice_info)
            else:
                slices_rel += file_slices_len

        raise IndexError

    def sequence_reseted(self):
        if self.cur['slices_ind'] == 0:
            return True

        cur_slices = self.__curr_slices()
        supposed_prev_beg = self.__cur_slice()[0] - self.n_frames
        prev_beg = cur_slices[self.cur['slices_ind'] - 1][0]
        if supposed_prev_beg != prev_beg:
            return True

        return False

    @staticmethod
    def from_file(filename, n_frames, batch_size):
        with open(filename, 'r') as file:
            info = yaml.load(file)
            file.close()
            return DataGenerator(info=info, n_frames=n_frames, batch_size=batch_size)


if __name__ == "__main__":
    with open("train.yaml", 'r') as stream:
        #try:
        data = yaml.load(stream)
        gen = DataGenerator(info=data, n_frames=512)
        i = 0

        # for x, y in gen:
        #     print(i)
        #     i += 1
        #     grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)
        #
        #     axes = plt.subplot(grid[:, 0])
        #     axes.imshow(x)
        #
        #     axes = plt.subplot(grid[:, 1])
        #     axes.imshow(y)
        #
        #     mng = plt.get_current_fig_manager()
        #     mng.window.state('zoomed')
        #
        #     plt.show()

        print(len(gen))

        # except yaml.YAMLError as exc:
        #     print(exc)
