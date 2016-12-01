from utils import parser
import numpy as np
import mido
import os


# get the path of the midi files
path = os.getcwd()+'/audio_resources'

# convert all the data into np and store it locally
parser.convert_array_to_nptensor(path)

