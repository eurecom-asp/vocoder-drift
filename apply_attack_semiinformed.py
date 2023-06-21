DESC="""
This thing takes a reverser checkpoint, a folder full of wavs, and tries to reverse their drift.
Will plunge me into depression. This time for real.
"""
import os
from argparse import ArgumentParser

import torch
import numpy as np

from atk_tools import SemiInformedReverser as Reverser, SupervisedDataset

###########

parser = ArgumentParser(description=DESC)

parser.add_argument('--reverser_path', required=True, type=str, help='Path to a state dict of a Reverser.')
parser.add_argument('--in_fold', required=True, type=str, help='Where to find wavs to reverse.')
parser.add_argument('--out_fold', required=True, type=str, help='Where to output the files in numpy xvector format.')
parser.add_argument('--device', required=False, type=str, default='cuda')

args = parser.parse_args()

device = args.device

if not os.path.exists(args.out_fold):
    os.makedirs(args.out_fold)

# Parse a list of ids from the files.
ids = [os.path.splitext(fn)[0].replace('_gen', '') for fn in os.listdir(args.in_fold) if '_gen.wav' in fn]

ds = SupervisedDataset(args.in_fold, ids, return_ids=True)
dl = torch.utils.data.DataLoader(
    ds,
    batch_size = 1,
    shuffle = False
)

reverser = Reverser().eval().to(device)
reverser.load_state_dict(torch.load(args.reverser_path, map_location='cuda'))

tot = len(dl)
with torch.no_grad():
    for i, (wav, y, uttids) in enumerate(dl):
        wav = wav.to(device)
        uttid = uttids[0] # batch size is hardcoded to 1

        reco_xv = reverser(wav, embeddings=True)

        fn = f'{uttid}_gen.xvector'
        fp = os.path.join(args.out_fold, fn)
        print(f'[{i+1}/{tot}] Saving to {fp}')
        np.save(fp, reco_xv.cpu().numpy())

    print('Done.')