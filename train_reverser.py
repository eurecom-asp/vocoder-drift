DESC = """
Simple script to train a reverser. This is deliberately quick and dirty, like every single thing I've written in the past 3 years.
(with a few notable exceptions)
"""
from argparse import ArgumentParser

import torch
import yaml


from atk_tools import Reverser, MatchingDataset, pad_to_longest, MetricTracker, inference


parser = ArgumentParser(description=DESC)

parser.add_argument('--config', required=False, type=str, default='config/config.yaml', help='Path of YAML config file')
parser.add_argument('--wav_root', required=True, type=str, help='Root of the wav files.')
parser.add_argument('--xv_root', required=True, type=str, help='Root of xv (pre-vocoder) files to use as targets for the reverser.')
parser.add_argument('--ids_list', required=True, type=str, help='Text list with all the IDS, one per line.')
parser.add_argument('--ids_list_val', required=False, default=None, help='If given, use this id list as validation data.')
parser.add_argument('--log_path', required=False, type=str, default='training.log', help='Will save a log file with training steps in it at this location.')
parser.add_argument('--tb_path', required=False, type=str, default='tb', help='Will save a tensorboard log file at this location.')
parser.add_argument('--checkpoint_path', required=False, type=str, default='checkpoints', help='Where to save model checkpoints.')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, yaml.CLoader)

device = config['device']
epochs = config['epochs']

# read the fucking file
with open(args.ids_list, 'r') as f:
    ids = [line.strip() for line in f.readlines()]

ds = MatchingDataset(args.wav_root, args.xv_root, ids)
dl = torch.utils.data.DataLoader(
    ds,
    batch_size = config['batch_size'],
    shuffle = True,
    collate_fn = pad_to_longest,
    drop_last = True
    )

if args.ids_list_val is not None:
    with open(args.ids_list_val, 'r') as fv:
        ids_val = [line.strip() for line in fv.readlines()]
    
    ds_val = MatchingDataset(args.wav_root, args.xv_root, ids_val)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size = 1,
        shuffle = False
    )

reverser = Reverser()
reverser.train()
reverser.to(device)

optimizer = torch.optim.Adam(reverser.parameters(), lr=config['lr'])

cos_sim = torch.nn.CosineSimilarity()
criterion = lambda inputs, target : (1 - cos_sim(inputs, target)).mean()

tracker = MetricTracker(args.log_path, args.tb_path)

tot = len(dl)
iteration = 0
min_val_loss = 2

for epoch in range(epochs):
    tracker.genericLog(f'EPOCH [{epoch+1}/{epochs}]')
    for i, (wav, xv) in enumerate(dl):
        wav = wav.to(device)
        xv = xv.to(device)

        optimizer.zero_grad()
        
        xv_p = reverser(wav)
        loss = criterion(xv_p, xv)
        
        loss.backward()
        optimizer.step()

        if iteration % config['print_rate']  == 0:
            tracker.update('loss', loss.item(), tb_iter=iteration)
            print(f'[{i+1}/{tot}] ', end='')
            tracker.display()
        
        # validation every 200 iterations
        if iteration % config['val_rate'] == 0 or i == tot-1:
            reverser.eval()
            
            mean_loss = inference(reverser, dl_val, criterion, device)
            tracker.genericLog(f'[VALIDATION iter {iteration}]')
            tracker.update('val_loss', mean_loss, tb_iter=iteration)
            tracker.display()

            reverser.train()

            if mean_loss < min_val_loss:
                tracker.genericLog(f'Found new best validation loss {mean_loss} at iteration {iteration}')
                min_val_loss = mean_loss
                torch.save(reverser.state_dict(), f'{args.checkpoint_path}/best_reverser_iter_{iteration}.pth')

        iteration += 1
