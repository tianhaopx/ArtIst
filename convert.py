from utils import parser
import numpy as np
import mido
import os

# mid = mido.MidiFile('1.mid')


path = os.getcwd()+'/audio_resources'
#midi_files  = parser.find_midi_path(path)

#for n in midi_files:
    #print(parser.midi_to_single_array(n).shape)

a = parser.convert_array_to_nptensor(path)
print(a[0][0])
