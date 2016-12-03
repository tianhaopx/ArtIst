import os
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage

# find all the midi files reture its file path
# and convert it into mido examples
def find_midi_path(path):
    files = os.listdir(path)
    files_path = [mido.MidiFile(path+'/'+n) if n.endswith('.mid') else None for n in files]
    while None in files_path:
        files_path.remove(None)
    return files_path


# we only need a array with notes on!
def getTicks(midi_files,_factor=12):
    ticks = []
    for mid in midi_files:
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0
            for message in track:
                if message.type in ['note_on','note_off']:
                    num_ticks += int(message.time/_factor)
            ticks.append(num_ticks)
    return max(ticks)

# first we get a array with notes on and off
def getNoteTimeOnOffArray(mid,_factor=12):
    note_time_onoff_array = []
    for track in mid.tracks:
        current_time = 0
        for message in track:
            if message.type in ['note_on','note_off']:
                current_time += int(message.time/_factor)
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
def createNetInputs(roll, target, seq_length=3072):
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

def midi_tensor(train_path,target_path):
    train_midi_files = find_midi_path(train_path)
    target_midi_files = find_midi_path(target_path)
    ticks = getTicks(train_midi_files)
    X = fromMidiCreatePianoRoll(train_midi_files,ticks)
    y = fromMidiCreatePianoRoll(target_midi_files,ticks)
    X, y = createNetInputs(X,y)
    np.save('data_x',X)
    np.save('data_y',y)
    print('Done!')


def NetOutToPianoRoll(network_output, threshold=0.1):
    piano_roll = []
    for i, timestep in enumerate(network_output):
        if np.amax(timestep) > threshold:
            pos = 0
            pos = np.argmax(timestep)
            timestep[:] = np.zeros(timestep.shape)
            timestep[pos] = 1
        else:
            timestep[:] = np.zeros(timestep.shape)
        piano_roll.append(timestep)

    return np.array(piano_roll)


def createMidiFromPianoRoll(piano_roll, directory=os.getcwd(), mel_test_file='_test', threshold=0.1, res_factor=12):

    ticks_per_beat = int(96/res_factor)
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
                    track.append(Message('note_on', note=k, velocity=100, time=time))
                if piano_roll[j+1, k] == 0 and piano_roll[j, k] == 1:
                    set_note += 1
                    track.append(Message('note_off', note=k, velocity=64, time=time))
    print(directory)
    mid.save('%s%s_th%s.mid' %(directory, mel_test_file, threshold))
    mid_files.append('%s.mid' %(mel_test_file))

    return
