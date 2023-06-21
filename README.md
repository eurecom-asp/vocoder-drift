# Vocoder drift in x-vector-based speaker anonymization
Repository containing code of [[1]](#mipa23).

## Installation
The required packages are in `requirements.txt`, as usual.

## Training the drift reversal network
Training code is provided in `train_reverser.py`. You will need:

- a folder containing anonymized waveforms, with the naming convention `<utterance_id>_gen.wav`;
- a folder containing pre-vococder x-vectors (named $\mathbf{x}_p$ in the paper), with the naming convention `<utterance_id>.xvector`, in a specific binary format (you can convert to this format from numpy array by using the `external.readwrite.write_raw_mat` function);
- a text file containing a list of utterance IDs used for training, one per line; and another containing the IDs used for validation, in the same format.

Some dummy files (10 utterances) are provided as an example in the `demo_files` folder. For more details about how to provide all of the above to the script, run the script itself with the help flag (`python train_reverser.py -h`).  
To launch a very short training (that uses the same 10 files for training and validation), run:
```
python train_reverser.py --wav_root demo_files/anon_wavs/ --xv_root demo_files/xvectors_anonymized/ --ids_list demo_files/demo_files.txt --ids_list_val demo_files/demo_files.txt
```

## Training the semi-informed attack
The procedure is mostly the same as the drift reversal, except that $\mathbf{x}_p$ vectors are not needed. A short training with the same 10 files as above can be run with:
```
python train_reverser_semiinformed.py --wav_root demo_files/anon_wavs/ --ids_list demo_files/demo_files.txt --ids_list_val demo_files/demo_files.txt
```

## Fine-tuning
Instead of training from scratch, it is possible to fine-tune the ECAPA models by providing a checkpoint path as value to the `eacapath` constructor argument when instantiating `atk_tools.Reverser` or `atk_tools.SemiInformedReverser`.
```
reverser = Reverser(ecapah='path/to/your/ecapa/checkpoint')
```
[The pretrained checkpoint I used is available for download at this link](https://nextcloud.eurecom.fr/s/3cEJqmLfrxyrXJw).

## Pre-trained reverser and semi-informed attacker
Both attacks are system-dependent, and you should therefore train them for the system you intend to work with.  
However, as a proof of concept (as in *"I swear I didn't make those numbers up"*), I am providing the checkpoints of the [drift reverser](https://nextcloud.eurecom.fr/s/T6TWJMWFBik52Pc) and the [semi-informed](https://nextcloud.eurecom.fr/s/t3DCJSpBrktGoa2) attacker for the HiFi-GAN version of the anonymization system.

They are pytorch state dictionaries and you load them as usual:
```
reverser = Reverser().eval().to('cuda')
reverser.load_state_dict(torch.load('/path/to/checkpoint'))
# the state dicts contain cuda tensors, remember you have to fiddle with map_location if you wanna have them on cpu
```

## Performing the attacks
Once you have a trained an attacker model, you can extract speaker embeddings from anonymized wavs to perform ASV.  
This time you don't need a file with wav paths, for the mere reason that I grew tired of having to generate it every time, so I just decided to parse a folder instead.  
As a result, all files to attack must be in the same folder. I'm sure you'll live.  
The files also have to follow the usual naming convention `<utterance_id>_gen.wav`.

For a dummy run of the drift reversal:
```
python apply_attack.py --reverser_path path/to/reverser/checkpoint --in_fold demo_files/anon_wavs/ --out_fold demo_files/out/ --target_fold demo_files/xvectors_anonymized/
```
...and the semi-informed:
```
python apply_attack_semiinformed.py --reverser_path path/to/checkpoint --in_fold demo_files/anon_wavs/ --out_fold demo_files/out/
```

## Where is the rest of the code?
To synthesize anonymized audio with different vocoders, I had to assemble together two code bases. The first one is the repository of [[2]](#miao22) (link in the paper), from which I took the HiFi-GAN version of the system; the second one is the [official NSF repository](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf), which I used to train the NSF and HiFi-NSF on the original feature representation of [[2]](#miao22).

If you really feel the irresistible urge to redo the exact experiments of the paper, I think the quickest way would be to head over those repositories and figure out how to make them work on your system by yourself. Providing my code would not be very useful, as it is quite hacky (since it is split across two repositories) and not too far off from qualifying as a disgrace to the discipline of tidy software writing.
Nevertheless, if you do need to do that, I'm willing to help - seriously. It's the least I can do. Do reach out, my email can be found at my [EURECOM page](https://www.eurecom.fr/en/people/panariello-michele).

You would be right in thinking this is not great for reproducibility. It's definitely a widespread issue in the speaker anonymization community - but we are aware of it, and resolved on trying to improve the situation.

# References
<span id="mipa23">[1]</span> *Vocoder drift in x-vector-based speaker anonymization*, https://arxiv.org/abs/2306.02892  
<span id="miao22">[2]</span> *Language-Independent Speaker Anonymization Approach using Self-Supervised Pre-Trained Models*, https://arxiv.org/abs/2202.13097