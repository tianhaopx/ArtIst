from utils import parser
import config
import os


conf = config.config()
sampling_freq = conf['sampling_freq']
sampling_block_size = conf['sampling_block_size']
max_seq_length = conf['max_seq_length']

path_to_audio_source = os.getcwd() + '/audio_sources'
file = parser.find_all_the_mp3(path_to_audio_source+'/mp3')
for n in file:
    parser.convert_mp3_to_wav(n,sampling_freq)


parser.convert_wav_files_to_nptensor(sampling_block_size,max_seq_length,'test')
