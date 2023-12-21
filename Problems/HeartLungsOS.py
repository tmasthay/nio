import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, norm, inputs_bool, device, which, mod, noise=0.0):
        self.file_data_testing = "data/BodyEIT_Comp_out.h5"
        self.file_data = "data/BodyEIT.h5"
        self.mod = mod
        self.noise = noise
        self.length = 512
        self.start = 0
        self.which = "training"
        self.reader = h5py.File(self.file_data, 'r')
        self.reader_testing = h5py.File(self.file_data_testing, 'r')
        self.mean_inp = torch.from_numpy(
            self.reader['mean_inp_fun'][:, :]
        ).type(torch.float32)
        self.mean_out = torch.from_numpy(
            self.reader['mean_out_fun'][:, :]
        ).type(torch.float32)
        self.std_inp = torch.from_numpy(self.reader['std_inp_fun'][:, :]).type(
            torch.float32
        )
        self.std_out = torch.from_numpy(self.reader['std_out_fun'][:, :]).type(
            torch.float32
        )

        self.mean_inp = np.delete(self.mean_inp, [16], axis=0)
        self.mean_inp = np.delete(self.mean_inp, [16], axis=1)

        self.std_inp = np.delete(self.std_inp, [16], axis=0)
        self.std_inp = np.delete(self.std_inp, [16], axis=1)

        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            self.inp_dim_branch = 32
            self.n_fun_samples = 20
        else:
            self.inp_dim_branch = 32
            self.n_fun_samples = 32

        self.norm = norm
        self.inputs_bool = inputs_bool

        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = self.reader_testing[self.which][
            'sample_' + str(index + self.start)
        ]["input"][:]
        inputs = np.delete(inputs, [16], axis=0)
        inputs = np.delete(inputs, [16], axis=1)
        inputs = torch.from_numpy(inputs).type(torch.float32)
        labels = torch.from_numpy(
            self.reader_testing[self.which][
                'sample_' + str(index + self.start)
            ]["output"][:]
        ).type(torch.float32)
        inputs = inputs * (1 + self.noise * torch.randn_like(inputs))
        if self.norm == "norm":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "norm-inp":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = (
                2
                * (labels - self.min_model)
                / (self.max_model - self.min_model)
                - 1.0
            )
        elif self.norm == "norm-out":
            inputs = (
                2 * (inputs - self.min_data) / (self.max_data - self.min_data)
                - 1.0
            )
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "minmax":
            inputs = (
                2 * (inputs - self.min_data) / (self.max_data - self.min_data)
                - 1.0
            )
            labels = (
                2
                * (labels - self.min_model)
                / (self.max_model - self.min_model)
                - 1.0
            )
        elif self.norm == "none":
            inputs = inputs
            labels = labels
        else:
            raise ValueError()

        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            inputs = inputs.view(1, 32, 32)
        else:
            inputs = inputs.permute(1, 0)

        return inputs, labels

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm" or self.norm == "norm-out":
            return tensor * (self.std_out + 1e-16).to(
                self.device
            ) + self.mean_out.to(self.device)
        elif self.norm == "none":
            return tensor
        else:
            return (self.max_model - self.min_model) * (
                tensor + torch.tensor(1.0, device=self.device)
            ) / 2 + self.min_model

    def get_grid(self):
        grid = torch.from_numpy(self.reader['grid'][:, :]).type(torch.float32)

        return grid.unsqueeze(0)
