DESC="""
This thing takes a reverser checkpoint, a folder full of wavs, and tries to reverse their drift.
Will plunge me into depression.
"""
import os
from argparse import ArgumentParser

import torch
import numpy as np

from atk_tools import Reverser, MatchingDataset

###########

parser = ArgumentParser(description=DESC)

parser.add_argument('--reverser_path', required=True, type=str, help='Path to a state dict of a Reverser.')
parser.add_argument('--in_fold', required=True, type=str, help='Where to find wavs to reverse.')
parser.add_argument('--out_fold', required=True, type=str, help='Where to output the files in numpy xvector format.')
parser.add_argument('--target_fold', required=True, type=str, help='Where the ideal target of the reverser are. Only used to compute some loss.')
parser.add_argument('--device', required=False, type=str, default='cuda')

args = parser.parse_args()

device = args.device

if not os.path.exists(args.out_fold):
    os.makedirs(args.out_fold)

# Parse a list of ids from the files.
ids = [os.path.splitext(fn)[0].replace('_gen', '') for fn in os.listdir(args.in_fold) if '_gen.wav' in fn]

ds = MatchingDataset(args.in_fold, args.target_fold, ids, return_id=True)
dl = torch.utils.data.DataLoader(
    ds,
    batch_size = 1,
    shuffle = False
)

reverser = Reverser().eval().to(device)
reverser.load_state_dict(torch.load(args.reverser_path))

cos_sim = torch.nn.CosineSimilarity()
criterion = lambda inputs, target : (1 - cos_sim(inputs, target)).mean() # this mean() is useless here, w/e

# finire qui
losses = []
tot = len(dl)
with torch.no_grad():
    for i, (wav, xv, uttids) in enumerate(dl):
        wav = wav.to(device)
        xv = xv.to(device)
        uttid = uttids[0] # batch size is hardcoded to 1
        
        reco_xv = reverser(wav)

        fn = f'{uttid}_gen.xvector'
        fp = os.path.join(args.out_fold, fn)
        print(f'[{i+1}/{tot}] Saving to {fp}')
        np.save(fp, reco_xv.cpu().numpy())

        # Computing and displaying loss for the hell of it
        loss = criterion(xv, reco_xv).item()
        print(f'(loss: {loss:.4})')
        losses.append(loss)

    # also display mean loss at the end of it
    mean_loss = np.mean(losses)
    print(f'Mean loss: {mean_loss:.4}')
    print('Does that suck?')
    print('(probably)')