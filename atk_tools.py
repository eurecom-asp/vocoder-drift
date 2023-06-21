import torch
import numpy as np
import librosa
from math import ceil
from tqdm import tqdm

from external.ecapa_tdnn_sb import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from external.AdMSLoss import AdMSoftmaxLoss


from external.readwrite import read_raw_mat

# for the tracker farlocco
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import json
import os


class Reverser(torch.nn.Module):
    """
    Some model that takes a waveform and spits out an x-vector, essentially.
    Has a nome del cazzo.
    """
    def __init__(self, ecapath=None):
        super().__init__()

        self.xv_model = ECAPA_TDNN(80, lin_neurons=192, out_neurons=5994)
        if ecapath is not None:
            ecapa_check = torch.load(ecapath, map_location='cpu')
            self.xv_model.load_state_dict(ecapa_check)
        self.xv_model.requires_grad_(True)

        # Whatever damn normalization was done originally
        self.fbank = Fbank(n_mels=80)
        # norm each sentence along each dim -mean/var
        self.mean_var_norm = InputNormalization(norm_type='sentence', std_norm=False)

    def forward(self, x):
        """
        x is a batch of torch audios of shape (B, L).
        """
        mel = self.fbank(x)
        mel_norm = self.mean_var_norm(mel, torch.ones(mel.shape[0]))
        xv, _ = self.xv_model(mel_norm)
        xv = xv.squeeze()
        return xv

class SemiInformedReverser(torch.nn.Module):
    """
    Some model that takes a waveform and spits out an x-vector, essentially.
    Has a nome del cazzo.
    """
    def __init__(self, n_classes=921, ecapath=None):
        super().__init__()

        self.xv_model = ECAPA_TDNN(80, lin_neurons=192, out_neurons=5994)
        
        if ecapath is not None:
            ecapa_check = torch.load(ecapath, map_location='cpu')
            self.xv_model.load_state_dict(ecapa_check)
        self.xv_model.requires_grad_(True)

        # Whatever damn normalization was done originally
        self.fbank = Fbank(n_mels=80)
        # norm each sentence along each dim -mean/var
        self.mean_var_norm = InputNormalization(norm_type='sentence', std_norm=False)
        self.amm_loss = AdMSoftmaxLoss(192, n_classes, s=30, m=0.2) # same as ecapa paper, 921 is the amount of speakers in libri-train-clean-360

    def forward(self, x, y=None, embeddings=False):
        """
        x is a batch of torch audios of shape (B, L).
        """
        if embeddings == False and y == None:
            raise ValueError("If embeddings=False (default value), the model assumes you are training. But you set y=None. This is not possible. Please provide y or set embeddings=True (the latter implies you are testing).")
        mel = self.fbank(x)
        mel_norm = self.mean_var_norm(mel, torch.ones(mel.shape[0]))
        xv, _ = self.xv_model(mel_norm)
        xv = xv.squeeze(dim=1)

        if embeddings:    
            return xv
        else:
            loss = self.amm_loss(xv, y)
            return loss

        


class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, wavs_root, xvs_root, ids, return_id=False) -> None:
        """
        Args:
            wavs_path: folder where to find input wave files. Should be raw audio.
                Names will be used to match them to their target xvectors.
                Will automatically add '_gen' in the filename.
            xvs_path: folder where to find the target xvectors in raw format (<uttid>.xvector).
            ids: the list of ids to look for. Shold be a list of strings.
            return_id: if True, return the id of the utterance as well.

        Will return stuff in the format (waveform, xv), where waveform is of shape (L,) and xv is (192,).
        If return_id is True, will instead return (waveform, xv, id).
        """
        self.wavs_root = wavs_root
        self.xvs_root = xvs_root
        self.return_id = return_id

        # The data will be stored as a dict of format
        # {uttid: (waveform_path, xvector_path)}
        id_from_fn = lambda fn: os.path.splitext(fn.replace('_gen', ''))[0]
        
        self.meta = {}
        for id in ids:
            wavp = os.path.join(self.wavs_root, f'{id}_gen.wav')

            assert id not in self.meta, f"Error, id {id} appears to have been read already."

            self.meta[id] = []
            self.meta[id].append(wavp)

            xvp = os.path.join(self.xvs_root, f'{id}.xvector')
            self.meta[id].append(xvp)

        # ok we are done, fuck my life. Now we just need to turn that into something iterable
        # like [(id, wav_path, xv_path)]
        self.meta = [(id, v[0], v[1]) for id, v in self.meta.items()]

    def __getitem__(self, idx):
        uttid, wavp, xvp = self.meta[idx]
        wav, _ = librosa.load(wavp, sr=None)
        # xv = np.squeeze(np.load(xvp))
        xv = np.squeeze(read_raw_mat(xvp, 192))
        wav = torch.tensor(wav)
        xv = torch.tensor(xv)
        if self.return_id:
            return wav, xv, uttid
        else:
            return wav, xv

    def __len__(self):
        return len(self.meta)
    
    # ta-daaa, cazzo

class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, wavs_root, ids, return_ids=False) -> None:
        self.wavs_root = wavs_root
        self.return_ids = return_ids

        # The data will be stored as a dict of format
        # {uttid: (waveform_path, xvector_path)}

        # get the speaker codes
        spk_codes = list(set([uttid.split('-')[0] for uttid in ids]))
        spk_codes = {k:v for v, k in enumerate(spk_codes)} # now you have a dict in the format spk_codes['928'] = 0
        
        self.meta = {}
        for id in ids:
            wavp = os.path.join(self.wavs_root, f'{id}_gen.wav')

            assert id not in self.meta, f"Error, id {id} appears to have been read already."

            self.meta[id] = []
            self.meta[id].append(wavp)

            spk_id = id.split('-')[0]
            spk_code = spk_codes[spk_id]

            self.meta[id].append(spk_code)

            # actually I also need the id

        # ok we are done, fuck my life. Now we just need to turn that into something iterable
        # like [(wav_path, spk_code, uttid)]
        # self.meta = list(self.meta.values()) # this is old, don't mind this
        self.meta = [(v[0], v[1], id) for id, v in self.meta.items()]
        

    def __getitem__(self, idx):
        wavp, spkcode, uttid = self.meta[idx]
        wav, _ = librosa.load(wavp, sr=None)
        wav = torch.tensor(wav)
        if self.return_ids:
            return wav, spkcode, uttid
        else:
            return wav, spkcode

    def __len__(self):
        return len(self.meta)


def pad_to_longest(samples):
    """
    Aw shit, here we go again.
    collate_fn to pad to the longest vector in the sequence in 'replicate mode',
    i.e. repeat the short waveforms until they match the longest one.
    """
    wavs, xvs = zip(*samples)
    max_length = max([len(wav) for wav in wavs])
    # add the padding
    padded_wavs = []
    for wav in wavs:
        # do some fucking bullshit to make it so that you can circular-pad in torch
        needed_samples = max_length - len(wav)
        num_repetitions = ceil(needed_samples / len(wav))
        padded_wav = wav.repeat(num_repetitions + 1)[:max_length]
        padded_wavs.append(padded_wav)
    wavs_batch = torch.stack(padded_wavs)
    xvs = torch.stack(xvs)
    return wavs_batch, xvs

def pad_to_longest_supervised(samples):
    """
    Aw shit, here we go again.
    collate_fn to pad to the longest vector in the sequence in 'replicate mode',
    i.e. repeat the short waveforms until they match the longest one.
    This is for the supervised setting where the label is an id.
    """
    wavs, spkids = zip(*samples)
    max_length = max([len(wav) for wav in wavs])
    # add the padding
    padded_wavs = []
    for wav in wavs:
        # do some fucking bullshit to make it so that you can circular-pad in torch
        needed_samples = max_length - len(wav)
        num_repetitions = ceil(needed_samples / len(wav))
        padded_wav = wav.repeat(num_repetitions + 1)[:max_length]
        padded_wavs.append(padded_wav)
    wavs_batch = torch.stack(padded_wavs)
    spkids = torch.LongTensor(spkids)
    return wavs_batch, spkids


def inference(model, dl_test, criterion, device):
    """
    This expects the model to be in eval state.
    """
    tot_test = len(dl_test)
    
    losses = []
    with torch.no_grad():
        for wav, xv in tqdm(dl_test, desc='Validation'):
            wav = wav.to(device)
            xv = xv.to(device)

            xv_reco = model(wav)

            test_loss = criterion(xv_reco, xv).item()
            losses.append(test_loss)

        mean_loss = np.mean(losses)

    return mean_loss

def inference_supervised(model, dl_test, device):
    """
    This expects the model to be in eval state.
    """
    
    losses = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for wav, spkids in tqdm(dl_test, desc='Validation'):
            wav = wav.to(device)
            spkids = spkids.to(device)

            test_loss = model(wav, y=spkids).item()
            losses.append(test_loss)

            embeds = model(wav, embeddings=True)
            scores = model.amm_loss.fc(embeds)
            batch_preds = scores.argmax(dim=1).tolist()
            y_pred.extend(batch_preds)
            y_true.extend(spkids.to('cpu').tolist())

        mean_loss = np.mean(losses)
        accuracy = (np.array(y_pred) == np.array(y_true)).mean()

    return mean_loss, accuracy


class MetricTracker:
    """
    Keeps track of several metrics at the same time like a pro. Logs to tensorboard and also onto a file.
    Internally keeps a list for several possible parameters.
    :log_path: path where to put log file
    :tensorboard_path: path where to put the tensorboard file
    """
    def __init__(self, log_path, tensorboard_path=None):
        log_format = '%(asctime)s - %(message)s'
        datefmt = '%Y-%d-%m %I:%M:%S'
        # check the logging folder and create it if needed
        root_folder = os.path.dirname(log_path)
        if root_folder and not os.path.isdir(root_folder):
            os.mkdir(root_folder)
        handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=datefmt, handlers=handlers)

        if tensorboard_path is not None:
            self.writer = SummaryWriter(tensorboard_path)
        else:
            self.writer = SummaryWriter()
        self.values = {}
        self.tb_iter = 0

    def genericLog(self, msg):
        """
        Logs a generic message on the file.
        :param msg: Message to log
        """
        logging.info(msg)

    def update(self, name, new_value, tb_iter=None):
        """
        Appends to the list a certain value of a parameter.
        :param name: name of the parameter to update
        :param new_value: value to append to the list of values of that parameter
        :param tb_iter: tensorboard iteration number. If None, no value is logged to tensorboard
        """
        # Append to the internally tracked values
        if name not in self.values:
            self.values[name] = [new_value]
        else:
            self.values[name].append(new_value)

        # Tensorboard update
        if tb_iter is None:
            tb_iter = self.tb_iter
            self.tb_iter += 1

        self.writer.add_scalar(name, new_value, tb_iter)

    def display(self, names=None, timestamp=-1):
        """
        Display the value of the given metrics at the given timestep.
        :param names: List of names of metrics. Will print a value form these metrics.
        :param timestamp: Timestamp of which to print the value. Defaults to -1 (last value added)
        """
        # By default, log the last of all values
        if names is None:
            msg = ' | '.join([f'{name}: {series[timestamp]:6.6}' for name, series in self.values.items()])
        else:
            msg = ' | '.join([f'{name}: {self.values[name][timestamp]:6.6}' for name in names])
        logging.info(msg)

    def save_as_json(self, file_save_path=None):
        """
        Dump all lists as a JSON file.
        :param file_save_path:
        """
        if file_save_path is None:
            file_save_path = '/content/runs/tracked_metrics.json'

        with open(file_save_path, 'w') as fp:
            json.dump(self.values, fp, indent=2)