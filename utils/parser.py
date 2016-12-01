import os
import numpy as np
import mido

# find all the midi files reture its file path
# and convert it into mido examples
def find_midi_path(path):
    files = os.listdir(path)
    files_path = [mido.MidiFile(path+'/'+n) if n.endswith('.mid') else None for n in files]
    while None in files_path:
        files_path.remove(None)
    return files_path

# all track transfer into one matrix
def midi_to_single_array(midi):
    """
    Convert MIDI file to a 2D NumPy ndarray (notes, timesteps).
    Put all the track into one ndarray
    """
    notes = 128  # there are total 128 notes in midi files
    tempo = 500000  # Assume same default tempo and 4/4 for all MIDI files.
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    # we use this part to store every notes
    velocities = np.zeros(notes)
    # and this store everything together
    sequence = []
    for m in midi:
        ticks = int(np.round(m.time / seconds_per_tick))
        ls = [velocities.copy()] * ticks
        sequence.extend(ls)
        if m.type == 'note_on':
            velocities[m.note] = m.velocity
        elif m.type == 'note_off':
            velocities[m.note] = 0
        else:
            continue
    single_array = np.array(sequence)
    return single_array



def midi_to_single_array1(midi):
    """
    Convert MIDI file to a 2D NumPy ndarray (notes, timesteps).
    Put all the track into (x,x,x)
    """
    notes = 128  # there are total 128 notes in midi files
    tempo = 500000  # Assume same default tempo and 4/4 for all MIDI files.
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    # we use this part to store every notes
    velocities = np.zeros(notes)
    # and this store everything together
    sequence = []
    for m in midi:
        ticks = int(np.round(m.time / seconds_per_tick))
        ls = velocities.copy() * ticks
        if sequence == []:
            sequence = ls.reshape([1,128,1])
        else:
            sequence = np.append(sequence,ls.reshape([1,128,1]),axis=0)
        if m.type == 'note_on':
            velocities[m.note] = m.velocity
        elif m.type == 'note_off':
            velocities[m.note] = 0
        else:
            continue
    return sequence




def midi_to_single_array2(midi):
    pitch = {}
    message = np.zeros([1,4])
    sequence = []
    for n in midi:
        if n.type == 'pitchwheel':
            pitch[n.channel] = n.pitch
        elif n.type in ['note_off','note_on']:
            #print(n.note,n.time,pitch[n.channel],n.velocity)
            ls = message.copy()
            if n.type == 'note_on':
                ls[0] = [n.note, n.time, n.velocity, pitch[n.channel]]
            else:
                ls[0] = [n.note, n.time, 0, pitch[n.channel]]
            if sequence == []:
                sequence = ls.reshape([1,1,4])
            else:
                sequence = np.append(sequence,ls.reshape([1,1,4]),axis=0)
    return sequence



# To do
# Seperate different tracks into different array




def convert_array_to_nptensor(path, max_files=20, out_file='out_put'):
    files = find_midi_path(path)
    chunks_X = None
    chunks_Y = None
    num_files = len(files)

    if(num_files > max_files):
        num_files = max_files

    for file_idx in range(num_files):
        file = files[file_idx]
        print('Processing: ', (file_idx+1),'/',num_files)
        print('File information: ', file)
        X = midi_to_single_array2(file)
        Y = X[1:]
        Y = np.append(Y,np.zeros([1,1,4]),axis=0)
        cur_seq = 0
        total_seq = len(X)
        print('Total sequence number:',total_seq)

        if chunks_X == None:
            chunks_X = X
        else:
            chunks_X = np.append(chunks_X, X, axis = 0)

        if chunks_Y == None:
            chunks_Y = Y
        else:
            chunks_Y = np.append(chunks_Y, Y, axis = 0)


    np.save('data_x', chunks_X)
    np.save('data_y', chunks_Y)
    print('Done!')
