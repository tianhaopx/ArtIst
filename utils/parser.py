# USING THIS FUNCTION IN THE ROOT OF YOUR PROJECT
import os
import scipy.io.wavfile as wav
from pipes import quote
import numpy as np
import config


# find all the mp3 files reture its file path
def find_all_the_mp3(path):
    files = os.listdir(path)
    files_path = [path+'/'+n for n in files]
    return files_path

# We first convert a MP3 file into WAV
def convert_mp3_to_wav(file_path, sample_frequency):
    path = '/'.join(file_path.split('/')[0:-2])
    filename = file_path.split('/')[-1]
    temp_file_path = path+'/tmp/'
    new_file_path = path+'/wav/'
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)
    temp_file_name = temp_file_path + filename
    new_file_name = new_file_path + filename.split('.')[0]+'.wav'
    sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
    cmd = 'lame -a -m m {0} {1}'.format(quote(file_path), quote(temp_file_name))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(temp_file_name), quote(new_file_name), sample_freq_str)
    os.system(cmd)
    return new_file_name



# read the WAV file into np
# Open a WAV file
# Return the sample rate (in samples/sec) and data from a WAV file.
def read_wav_as_np(filename):
    path = os.getcwd()+'/audio_sources/wav/'
    files = os.listdir(path)
    if filename not in files:
        return "Cannot find files"
    data = wav.read(path+filename)
    np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
    np_arr = np.array(np_arr)
    return np_arr, data[0]


# A piece of music can be interpret as the combination of
# Time Domain and Frequence Domain
# convert np into sample block
def convert_np_audio_to_sample_blocks(song_np, block_size):
    block_lists = []
    total_samples = song_np.shape[0]
    num_samples_so_far = 0
    while(num_samples_so_far < total_samples):
        block = song_np[num_samples_so_far:num_samples_so_far+block_size]
        if(block.shape[0] < block_size):
            padding = np.zeros((block_size - block.shape[0],))
            block = np.concatenate((block, padding))
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists



# https://zhuanlan.zhihu.com/p/19763358?columnSlug=wille
# read about this article!
# this function we use fast fourier to transfer time domain block
# 总的来说，傅里叶变换告诉你，在一个整体的波形中，每一个单独的“音符”（正弦曲线或是圆圈）的比例
# 对于每个音频片段，傅里叶变换将音频波形分解为它的成分音符并且保存下来，从而代替存储原始波形。
# http://blog.jobbole.com/51301/
def time_blocks_to_fft_blocks(blocks_time_domain):
    fft_blocks = []
    for block in blocks_time_domain:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)
    return fft_blocks


# Loading the np and fft
def load_training_example(filename, block_size=2048, useTimeDomain=False):
    data, bitrate = read_wav_as_np(filename)
    # x_t is input and y_t is output
    x_t = convert_np_audio_to_sample_blocks(data, block_size)
    # except first row
    y_t = x_t[1:]
    #Add special end block composed of all zeros
    y_t.append(np.zeros(block_size))

    if useTimeDomain:
        return x_t, y_t

    X = time_blocks_to_fft_blocks(x_t)
    Y = time_blocks_to_fft_blocks(y_t)
    return X, Y

# transfer the file into npy
# before you run this function
# you need to make sure there are WAV files
def convert_wav_files_to_nptensor(block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
    files = []
    directory = os.getcwd()+'/audio_sources/wav'

    # read WAV file from directory
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(file)
    chunks_X = []
    chunks_Y = []
    num_files = len(files)

    # limited the source files number
    if(num_files > max_files):
        num_files = max_files


    for file_idx in range(num_files):
        file = files[file_idx]
        print('Processing: ', (file_idx+1),'/',num_files)
        print('Filename: ', file)
        X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
        cur_seq = 0
        total_seq = len(X)
        print('Total sequence number:',total_seq)
        print('Max sequence length:',max_seq_len)
        while cur_seq + max_seq_len < total_seq:
            chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
            cur_seq += max_seq_len


    num_examples = len(chunks_X)
    num_dims_out = block_size * 2
    if(useTimeDomain):
        num_dims_out = block_size
    out_shape = (num_examples, max_seq_len, num_dims_out)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)
    for n in range(num_examples):
        for i in range(max_seq_len):
            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_Y[n][i]
        print('Saved example ', (n+1), ' / ',num_examples)
    print('Flushing to disk...')

    mean_x = x_data.mean(axis = (0,1)) # Mean of all data
    std_x = x_data.std(axis = (0,1)) # Std of all data
    std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny


    x_data[:][:] -= mean_x #Mean 0
    x_data[:][:] /= std_x #Variance 1
    y_data[:][:] -= mean_x #Mean 0
    y_data[:][:] /= std_x #Variance 1

    np.save(out_file+'_mean', mean_x)
    np.save(out_file+'_var', std_x)
    np.save(out_file+'_x', x_data)
    np.save(out_file+'_y', y_data)
    print('Done!')


def convert_sample_blocks_to_np_audio(blocks):
    song_np = np.concatenate(blocks)
    return song_np




# convert the fast fourier block into the original time domain block
def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        '''
        num_elems = block.shape[0]/2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
        '''
        time_blocks.append(np.fft.ifft(block))
    return time_blocks


# convert np into wav file
def write_np_as_wav(X, sample_rate, filename):
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    wav.write(filename, sample_rate, Xnew)
    return


def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
    if useTimeDomain:
        time_blocks = generated_sequence
    else:
        time_blocks = fft_blocks_to_time_blocks(generated_sequence)
        print(time_blocks)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, sample_frequency, filename)
    return

def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    for i in indices:
        chunks = []
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
        save_generated_example(filename+str(i)+'.wav', chunks,useTimeDomain=useTimeDomain)



# 44100 frequence
# block size 2048
# convert_mp3_to_wav('1.mp3',44100)
# a = read_wav_as_np('1.wav')
# b = convert_np_audio_to_sample_blocks(a[0],2048)
# c = b[1:]
# c.append(np.zeros(2048))
# print(c)
# print(load_training_example('test.wav')[1])

# convert_wav_files_to_nptensor(2048,20,'test')
'''
conf = config.config()
sampling_freq = conf['sampling_freq']
path_to_audio_source = os.getcwd() + '/audio_sources'
files = find_all_the_mp3(path_to_audio_source+'/mp3')
for n in files:
    convert_mp3_to_wav(n,sampling_freq)
'''
