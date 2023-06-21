DESC = """
Simple script to train a semi-informed attacker on an anonymization system.
This is also deliberately quick and dirty, because conference deadlines don't give a damn about fancy coding. Unfortunately.
"""
from argparse import ArgumentParser

import torch
import yaml

from atk_tools import SemiInformedReverser as Reverser, SupervisedDataset, pad_to_longest_supervised, MetricTracker, inference_supervised


parser = ArgumentParser(description=DESC)

parser.add_argument('--config', required=False, type=str, default='config/config.yaml', help='Path of YAML config file')
parser.add_argument('--wav_root', required=True, type=str, help='Root of the wav files.')
parser.add_argument('--ids_list', required=True, type=str, help='Text list with all the IDS, one per line.')
parser.add_argument('--ids_list_val', required=False, default=None, help='If given, use this id list as validation data.')
parser.add_argument('--log_path', required=False, type=str, default='training.log')
parser.add_argument('--tb_path', required=False, type=str, default='tb')
parser.add_argument('--checkpoint_path', required=False, type=str, default='checkpoints')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, yaml.CLoader)

device = config['device']
epochs = config['epochs']

# read the f file
with open(args.ids_list, 'r') as f:
    ids = [line.strip() for line in f.readlines()]


ds = SupervisedDataset(args.wav_root, ids)
dl = torch.utils.data.DataLoader(
    ds,
    batch_size = config['batch_size'],
    shuffle = True,
    collate_fn = pad_to_longest_supervised,
    drop_last = True
    )

if args.ids_list_val is not None:
    with open(args.ids_list_val, 'r') as fv:
        ids_val = [line.strip() for line in fv.readlines()]
    
    ds_val = SupervisedDataset(args.wav_root, ids_val)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size = 1,
        shuffle = False
    )

reverser = Reverser()
reverser.train()
reverser.to(device)

optimizer = torch.optim.SGD(reverser.parameters(), lr=config['lr'], momentum=config['momentum'])

# This procedure below is an attempt to mimick the kaldi training procedure as closely as possible (or at least that's how I understand it)
tot_iters = len(dl)
final_lr_scheduler_iter = int(tot_iters * 0.95) # we linearly diminish the learning rate until *almost* the end of training (but leave some final iters with constant LR)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=final_lr_scheduler_iter)
tracker = MetricTracker(args.log_path, args.tb_path)

iteration = 0
max_val_acc = 0
for epoch in range(epochs):
    tracker.genericLog(f'EPOCH [{epoch+1}/{epochs}]')
    for i, (wav, spkids) in enumerate(dl):
        wav = wav.to(device)
        spkids = spkids.to(device)

        optimizer.zero_grad()
        
        loss = reverser(wav, y=spkids)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % config['print_rate']  == 0:
            tracker.update('loss', loss.item(), tb_iter=iteration)
            tracker.update('lr', scheduler.get_last_lr()[0], tb_iter=iteration)
            print(f'[{i+1}/{tot_iters}] ', end='')
            tracker.display()
        
        # validation every 200 iterations
        if iteration % config['val_rate'] == 0 or i == tot_iters-1:
            reverser.eval()
            
            mean_loss, accuracy = inference_supervised(reverser, dl_val, device)
            tracker.genericLog(f'[VALIDATION iter {iteration}]')
            tracker.update('val_loss', mean_loss, tb_iter=iteration)
            tracker.update('val_accuracy', accuracy, tb_iter=iteration)
            tracker.display()

            reverser.train()

            if accuracy > max_val_acc:
                tracker.genericLog(f'Found new best validation accuracy {accuracy} at iteration {iteration}')
                max_val_acc = accuracy
                torch.save(reverser.state_dict(), f'{args.checkpoint_path}/best_accuracy_reverser_iter_{iteration}.pth')

        iteration += 1
