from utils import parser
import os


# get the path of the midi files
# return the list of the midi file parse by mido
train_path = os.getcwd()+'/audio_resources/train'
target_path = os.getcwd()+'/audio_resources/target'

# use the function in the parser to transfer midi file into 3d tensor
# the tensor looks like
# (file, timestep, note)



# parser.midi_tensor(train_path,target_path)
parser.New_midi_tensor()
