from utils import parser
import numpy as np
import mido
import os


# get the path of the midi files
train_path = os.getcwd()+'/audio_resources/train'
target_path = os.getcwd()+'/audio_resources/target'

# use the function in the parser to transfer midi file into tensor
#parser.midi_tensor(train_path,target_path)
parser.New_midi_tensor()
