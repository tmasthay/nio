import NIOModules
import h5py
import torch
import hydra
from omegaconf import DictConfig
from masthay_helpers.global_helpers import format_with_black, subdict
from masthay_helpers.typlotlib import plot_tensor2d_subplot
import numpy as np
import time


@hydra.main(config_path="config", config_name="helm", version_base=None)
def main(cfg: DictConfig):
    cfg = cfg[cfg.case]
    model = torch.load(cfg.model)

    with h5py.File(cfg.data, "r") as f:
        train, test, val = f['testing'], f['training'], f['validation']

        def torchify(x, dim=0):
            return torch.stack([torch.from_numpy(e) for e in x], dim=dim)
        
        def get_samples(d):
            keys = np.random.choice(
                list(d.keys()), size=cfg.plot.samples, replace=False
            )
            input_data = [d[k]['input'][:] for k in keys]
            output = [d[k]['output'][:] for k in keys]

            pred_start = time.time()
            pred = []
            grid = torch.from_numpy(
                np.array(f['grid'][:], dtype=np.float32)
            ).to('cuda')
            N = len(input_data)
            for i, e in enumerate(input_data):
                e = torch.from_numpy(
                    np.array(e.reshape(*cfg.shape), dtype=np.float32)
                ).to('cuda')
                pred.append(model(e, grid).cpu().detach().numpy())
            avg_pred_time = (time.time() - pred_start) / len(input_data)
            print(f'avg pred time: {avg_pred_time:.3f}')

            input_data = torchify(input_data, dim=-1)
            output = torchify(output, dim=-1)
            pred = torchify(pred, dim=-1).squeeze(0)
            return {'input': input_data, 'output': output, 'pred': pred}

        d = {
            'grid': torchify([f['grid'][:]]),
            'fun': [f['mean_inp_fun'][:], f['mean_out_fun'][:], f['std_inp_fun'][:], f['std_out_fun'][:]],
            'test': get_samples(test),
            'train': get_samples(train),
            'val': get_samples(val)
        }

        # d['grid'] = d['grid'].permute(3, 0, 1, 2)

        def plot_kw(key):
            return {**cfg.plot.defaults, **cfg.plot.get(key, {})}

        plot_tensor2d_subplot(
            tensor=d['grid'],
            name='grid',
            labels=[['grid', 'comp0', 'comp1', 'component']],
            **plot_kw('grid')
        )
        
        plot_tensor2d_subplot(
            tensor=d['fun'],
            name='fun',
            labels=[[e, 'comp0', 'comp1'] for e in ['mean_inp', 'mean_out', 'std_inp', 'std_out']],
            **plot_kw('fun')
        )
        
        for k in ['train', 'val', 'test']:
            plot_tensor2d_subplot(
                tensor=list(d[k].values()),
                name=k,
                labels=[[e, 'comp0', 'comp1'] for e in ['Input', 'Output', 'Pred']],
                **plot_kw(k)
            )



if __name__ == "__main__":
    main()
