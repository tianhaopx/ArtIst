from utils import parser
import numpy as np
import mido
import os


# get the path of the midi files
train_path = os.getcwd()+'/audio_resources/train'
target_path = os.getcwd()+'/audio_resources/target'
# convert all the data into np and store it locally
#parser.convert_array_to_nptensor(path)
train_midi_files = parser.find_midi_path(train_path)
target_midi_files = parser.find_midi_path(target_path)
parser.convert_array_to_nptensor1(train_midi_files, target_midi_files)
