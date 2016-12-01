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
                sequence = ls.reshape([1,4])
            else:
                sequence = np.append(sequence,ls.reshape([1,4]),axis=0)
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




# we only need a array with notes on!


def getTicks(midi_files):
    ticks = []
    for mid in midi_files:
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0
            for message in track:
                if message.type in ['note_on','note_off']:
                    num_ticks += int(message.time)
            ticks.append(num_ticks)

    return max(ticks)






# first we get a array with notes on and off
def getNoteTimeOnOffArray(mid):
    note_time_onoff_array = []
    for track in mid.tracks:
        current_time = 0
        for message in track:
            if message.type in ['note_on','note_off']:
                current_time += int(message.time)
                if message.type == 'note_on':
                    note_onoff = 1
                elif message.type == 'note_off':
                    note_onoff = 0
                else:
                    print("Error: Note Type not recognized!")

                note_time_onoff_array.append([message.note, current_time, note_onoff])
    return note_time_onoff_array

# now we filter the notes off
# and replace the on/off with the notes length
def getNoteOnLengthArray(note_time_onoff_array):
    note_on_length_array = []
    for i, message in enumerate(note_time_onoff_array):
        if message[2] == 1: #if note type is 'note_on'
            start_time = message[1]
            for event in note_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
                if event[0] == message[0] and event[2] == 0:
                    length = event[1] - start_time
                    break
            note_on_length_array.append([message[0], start_time, length])
    return note_on_length_array

# now we combine everything together
def fromMidiCreatePianoRoll(midi_files, ticks):
    num_files = len(midi_files)

    piano_roll = np.zeros((num_files, ticks, 128), dtype=np.float32)

    for i, mid in enumerate(midi_files):
        note_time_onoff = getNoteTimeOnOffArray(mid)
        note_on_length = getNoteOnLengthArray(note_time_onoff)
        for message in note_on_length:
            piano_roll[i, message[1]:(message[1]+int(message[2]/2)), message[0]] = 1
    return piano_roll


def createNetInputs(roll, target, seq_length=3072):
    #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
    #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
    #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars

    X = []
    y = []
    print('Lets start!')
    for i, song in enumerate(roll):
        print(i)
        pos = 0
        while pos+seq_length < song.shape[0]:
            print(pos+seq_length,song.shape[0])
            sequence = np.array(song[pos:pos+seq_length])
            X.append(sequence)
            y.append(target[i, pos+seq_length])
            pos += 1


    return np.array(X), np.array(y)




def convert_array_to_nptensor1(train_midi_files,target_midi_files):
    max_Ticks_train = getTicks(train_midi_files)
    max_Ticks_target = getTicks(target_midi_files)
    Train = fromMidiCreatePianoRoll(train_midi_files, max_Ticks_train)
    Target = fromMidiCreatePianoRoll(target_midi_files, max_Ticks_train)
    #Train, Target = createNetInputs(Train, Target)
    np.save('data_x', Train)
    np.save('data_y', Target)
    print('Done!')




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
