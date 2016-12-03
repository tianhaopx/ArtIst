import os
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage

# This seems to be a better version for us I think
# Afterall I don't know what hell is Resolution Factors
# this return all the ticks
# but somehow it runs faster than with the res_factor
# if use this one
# the getTicks res_factor has to be set to 1
def to_piano_roll(midi):
    """Convert MIDI file to a 2D NumPy ndarray (notes, timesteps)."""
    notes = 128
    tempo = 500000  # Assume same default tempo and 4/4 for all MIDI files.
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    velocities = np.zeros(notes)
    sequence = []
    for m in midi:
        if m.type in ['note_on','note_off']:
            ticks = int(np.round(m.time / seconds_per_tick))
            ls = [velocities.copy()] *ticks
            sequence.extend(ls)
            if m.type == 'note_on':
                velocities[m.note] = 1
            elif m.type == 'note_off':
                velocities[m.note] = 0
            else:
                continue
    piano_roll = np.array(sequence)
    return piano_roll


def New_fromMidiCreatePianoRoll(midi_files,ticks):
    num_files = len(midi_files)
    piano_roll = np.zeros((num_files, ticks, 128), dtype=np.float32)
    for i, mid in enumerate(midi_files):
        note_on_length = to_piano_roll(mid)
        piano_roll[i] = note_on_length[0:ticks]
    return piano_roll

def New_getTicks(midi_files):
    ticks = []
    for mid in midi_files:
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0
            for message in track:
                if message.type in ['note_on','note_off']:
                    num_ticks += int(message.time)
            if num_ticks != 0:
                ticks.append(num_ticks)
    return min(ticks)


def New_fromMidiCreatePianoRoll(midi_files,ticks):
    num_files = len(midi_files)
    piano_roll = np.zeros((num_files, ticks, 128), dtype=np.float32)

    for i, mid in enumerate(midi_files):
        note_on_length = to_piano_roll(mid)
        piano_roll[i] = note_on_length[0:ticks]
    return piano_roll


def New_midi_tensor():
    path1 = os.getcwd()+'/audio_resources/train'
    path2 = os.getcwd()+'/audio_resources/test'
    a = find_midi_path(path1)
    b = find_midi_path(path2)
    ticks = New_getTicks(a)
    X = New_fromMidiCreatePianoRoll(a,ticks)
    y = New_fromMidiCreatePianoRoll(b,ticks)
    print(X.shape)
    print(y.shape)
    np.save('data_x',X)
    np.save('data_y',y)
    print('Done!')

# find all the midi files reture its file path
# and convert it into mido examples
def find_midi_path(path):
    files = os.listdir(path)
    files_path = [mido.MidiFile(path+'/'+n) if n.endswith('.mid') else None for n in files]
    while None in files_path:
        files_path.remove(None)
    return files_path


# we only need a array with notes on!
def getTicks(midi_files,res_factor=12):
    ticks = []
    for mid in midi_files:
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0
            for message in track:
                if message.type in ['note_on','note_off']:
                    num_ticks += int(message.time/res_factor)
            ticks.append(num_ticks)
    return max(ticks)

# first we get a array with notes on and off
def getNoteTimeOnOffArray(mid,res_factor=12):
    note_time_onoff_array = []
    for track in mid.tracks:
        current_time = 0
        for message in track:
            if message.type in ['note_on','note_off']:
                current_time += int(message.time/res_factor)
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

# change seq_length!!!!!
# this part roll and target must have the same sample size?
# I'm not sure about this thing
def createNetInputs(roll, target, seq_length=1):
    #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
    #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
    #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars

    # i --- sample
    # song ------matrix
    X = []
    y = []
    for i, song in enumerate(roll):
        pos = 0
        while pos+seq_length < song.shape[0]:
            sequence = np.array(song[pos:pos+seq_length])
            X.append(sequence)
            y.append(target[i, pos+seq_length])
            pos += 1

    return np.array(X), np.array(y)

def midi_tensor(train_path,test_path):
    train_midi_files = find_midi_path(train_path)
    test_midi_files = find_midi_path(test_path)
    ticks_train = getTicks(train_midi_files)
    ticks_test = getTicks(test_midi_files)
    X = fromMidiCreatePianoRoll(train_midi_files,ticks)
    y = fromMidiCreatePianoRoll(test_midi_files,ticks_test)
    X, y = createNetInputs(X,y)
    np.save('data_x',X)
    np.save('data_y',y)
    print('Done!')

# this is wrong?
# I think it has to be changed
def NetOutToPianoRoll(network_output, threshold=0.1, chord_threshold=1e-5):
    piano_roll = []
    for i, timestep in enumerate(network_output):
        if np.amax(timestep) > threshold:
            pos = 0
            pos = np.argmax(timestep)
            # pos = np.argwhere(timestep >= np.amax(timestep)-chord_threshold)
            timestep[:] = np.zeros(timestep.shape)
            timestep[pos] = 1
        else:
            timestep[:] = np.zeros(timestep.shape)
        piano_roll.append(timestep)

    return np.array(piano_roll)

# so does it!
# if we use the old way
# this function will make the song extremely long
# gotta make some changes
def createMidiFromPianoRoll(piano_roll, directory=os.getcwd(), mel_test_file='_test', threshold=0.1):

    ticks_per_beat = int(180)
    mid = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    mid_files = []


    delta_times = [0]
    for k in range(piano_roll.shape[1]):#initial starting values
        if piano_roll[0, k] == 1:
            track.append(Message('note_on', note=k, velocity=100, time=0))
            delta_times.append(0)

    for j in range(piano_roll.shape[0]-1):#all values between first and last one
        set_note = 0 #Check, if for the current timestep a note has already been changed (set to note_on or note_off)

        for k in range(piano_roll.shape[1]):
            if (piano_roll[j+1, k] == 1 and piano_roll[j, k] == 0) or (piano_roll[j+1, k] == 0 and piano_roll[j, k] == 1):#only do something if note_on or note_off are to be set
                if set_note == 0:
                    time = j+1 - sum(delta_times)
                    delta_times.append(time)
                else:
                    time = 0

                if piano_roll[j+1, k] == 1 and piano_roll[j, k] == 0:
                    set_note += 1
                    track.append(Message('note_on', note=k, velocity=120, time=time))
                if piano_roll[j+1, k] == 0 and piano_roll[j, k] == 1:
                    set_note += 1
                    track.append(Message('note_off', note=k, velocity=120, time=time))

    mid.save('%s%s_th%s.mid' %(directory, mel_test_file, threshold))
    mid_files.append('%s.mid' %(mel_test_file))

    return
