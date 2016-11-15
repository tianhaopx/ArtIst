import numpy as np

#A very simple seed generator
#Copies a random example's first seed_length sequences as input to the generation algorithm
def generate_copy_seed_sequence(seed_length, training_data):
    num_examples = training_data.shape[0]
    example_len = training_data.shape[1]
    randIdx = np.random.randint(num_examples, size=1)[0]
    randSeed = np.concatenate(tuple([training_data[randIdx + i] for i in range(seed_length)]), axis=0)
    seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))

    return seedSeq


import numpy as np

#Extrapolates from a given seed sequence
def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
    seedSeq = seed.copy()
    output = []

    #The generation algorithm is simple:
    #Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
    #Step 2 - Concatenate X_n + 1 onto A
    #Step 3 - Repeat MAX_SEQ_LEN times
    for it in range(sequence_length):
        print(seedSeq.shape,'============')
        seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
        #Step 2. Append it to the sequence
        if it == 0:
            for i in range(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy())
        newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)

    #Finally, post-process the generated sequence so that we have valid frequencies
    #We're essentially just undo-ing the data centering process
    for i in range(len(output)):
        output[i] *= data_variance
        output[i] += data_mean
    return output
