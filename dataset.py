from torch.utils.data import Dataset
import numpy as np 
import torch
import librosa
import os
import pickle
import matplotlib.pyplot as plt

N_FFT = 256

class InstrumentsDataset(Dataset):
    def __init__(self, src='/home/nevronas/dataset/IRMAS-TrainingData', load=False, test=False):
        super(InstrumentsDataset, self).__init__()
        
        self.inst_spec_train = []
        #self.inst_label_train = []
        self.inst_spec_test = []
        #self.inst_label_test = []
        self.test = test

        insts = [
                'cel', # Cello
                'cla', # Clarinet
                'flu', # Flute
                'gac', # Acoustic guitar
                'gel', # Electric guitar
                'org', # Organ
                'pia', # Piano
                'sax', # Saxophone
                'tru', # Trumpet
                'vio'  # Violin
            ]

        if(load):
            label = 0
            for inst in insts:
                curr_inst_spec_train = []
                #curr_inst_label_train = []
                curr_inst_spec_test = []
                #curr_inst_label_test = []

                files = os.listdir(os.path.join(src, inst))
                j = 0
                for f in files:
                    print(f)
                    signal, fs = librosa.load(os.path.join(os.path.join(src, inst), f))
                    '''
                    D = librosa.stft(signal, N_FFT)
                    S, phase = librosa.magphase(D)
                    S = np.log1p(S)
                    S = S[:, 0:250]
                    S = np.expand_dims(S, 0)
                    if S.shape[2] < 250:
                        print("Chuck this sample!")
                        continue
                    '''
                    '''
                    S = librosa.feature.mfcc(y=signal,sr=fs,n_mfcc=20,n_fft=256)
                    list=[]
                    for i in range(0,120,20):
                        slice=S[:,i:i+20]
                        list.append(slice)
                    for i in range(6):
                        list[i] = np.expand_dims(list[i], 0)
                        print(list[i].shape)
                    if(j< int(0.8*len(files))):
                        
                            curr_inst_spec_train.extend(list)
                        #curr_inst_label_train.append(label)
                    else:
                        curr_inst_spec_test.extend(list)
                        #curr_inst_label_test.append(label)

                        #pickle1=stft(signal,512)
                        #pickle2=mfcc(signal)
                        #pickle=melspectrogram(signal)
                        '''
                    S=librosa.feature.melspectrogram(y=signal,sr=fs, n_mels=40, n_fft=1024)
                    #S=librosa.power_to_db(S,ref=np.max)
                    # print(S.shape)
                    #S= np.expand_dims(S,0)
                    list=[]
                    #list.append(S)
                    for i in range(0,130,20):
                        if(i==100):
                          slice=S[:,i:i+30]  
                          #t=np.zeros(:40,:40)
                          t[:,:30]=slice
                          slice=t
                          np.expand_dims(slice,0)
                          list.append(slice)  
                        else:
                          slice=S[:,i:i+40]
                          np.expand_dims(slice,0)
                          list.append(slice)     
                    #slice=S[:,0:40]
                    print(slice.shape)
                    print(list.shape)

                        # list.append(slice)
                    #list.append(np.expand_dims(slice,0))
                    # for i in range(2):
                    #      list[i]=np.expand_dims(list[i],0)
                    #      print(list[i].shape)
                    
                    if(j< int(0.8*len(files))):
                        for i in range(7):
                            curr_inst_spec_train.extend(list[i])
                        #curr_inst_label_train.append(label)
                    else:
                        for i in range(7):
                            curr_inst_spec_test.extend(list[i])
                        #curr_inst_label_test.append(label)
   
                    j = j+1
                file_handler = open('./pickle2/insts_train_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_train), file_handler)
                file_handler.close()
                file_handler = open('./pickle2/insts_test_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_test), file_handler)
                file_handler.close()
                label = label + 1

        else:

            for inst in insts:
                if(not self.test):
                    file_handler = open('./pickle2/insts_train_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_train.extend(curr_inst_spec)
                    #self.inst_label_train.extend(curr_inst_label)
                else:
                    file_handler = open('./pickle2/insts_test_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_test.extend(curr_inst_spec)
                    #self.inst_label_test.extend(curr_inst_label)

    def __len__(self):
        if(not self.test):
            return len(self.inst_spec_train)
        else:
            return len(self.inst_spec_test)

    def __getitem__(self, idx):
        if(not self.test):
            return self.inst_spec_train[idx]
        else:
            return self.inst_spec_test[idx]

# class InstrumentsDatasetMel(Dataset):
#     def __init__(self, src='/home/nevronas/dataset/IRMAS-TrainingData', load=False, test=False):
#         super(InstrumentsDatasetMel, self).__init__()
        
#         self.inst_spec_train = []
#        # self.inst_label_train = []
#         self.inst_spec_test = []
#         #self.inst_label_test = []
#         self.test = test

#         insts = [
#                 'cel', # Cello
#                 'cla', # Clarinet
#                 'flu', # Flute
#                 'gac', # Acoustic guitar
#                 'gel', # Electric guitar
#                 'org', # Organ
#                 'pia', # Piano
#                 'sax', # Saxophone
#                 'tru', # Trumpet
#                 'vio'  # Violin
#             ]

#         if(load):
#             label = 0
#             for inst in insts:
#                 curr_inst_spec_train = []
#                 #curr_inst_label_train = []
#                 curr_inst_spec_test = []
#                 #curr_inst_label_test = []

#                 files = os.listdir(os.path.join(src, inst))
#                 i = 0
#                 for f in files:
#                     print(f)
#                     signal, fs = librosa.load(os.path.join(os.path.join(src, inst), f))
#                     '''
#                     D = librosa.stft(signal, N_FFT)
#                     S, phase = librosa.magphase(D)
#                     S = np.log1p(S)
#                     S = S[:, 0:250]
#                     S = np.expand_dims(S, 0)
#                     if S.shape[2] < 250:
#                         print("Chuck this sample!")
#                         continue
                    
#                     S = librosa.feature.mfcc(signal)
#                     S = np.expand_dims(S, 0)
#                     '''
#                     D = librosa.stft(signal, N_FFT)
#                     S, phase = librosa.magphase(D)
#                     S = np.log1p(S)
#                     S = S[:, 0:250]
#                     S = np.expand_dims(S, 0)
#                     print(S.shape)
#                     if(i < int(0.8*len(files))):
#                         curr_inst_spec_train.append(S)
#                         #curr_inst_label_train.append(label)
#                     else:
#                         curr_inst_spec_test.append(S)
#                         #curr_inst_label_test.append(label)
#                     i = i+1
#                 file_handler = open('./pickle/instsmel_train_pickled_{}.dat'.format(inst), 'wb+')
#                 pickle.dump((curr_inst_spec_train), file_handler)
#                 file_handler.close()
#                 file_handler = open('./pickle/instsmel_test_pickled_{}.dat'.format(inst), 'wb+')
#                 pickle.dump((curr_inst_spec_test), file_handler)
#                 file_handler.close()
#                 label = label + 1

#         else:

#             for inst in insts:
#                 if(not self.test):
#                     file_handler = open('./pickle/instsmel_train_pickled_{}.dat'.format(inst), 'rb+')
#                     curr_inst_spec = pickle.load(file_handler)
#                     file_handler.close()
#                     self.inst_spec_train.extend(curr_inst_spec)
#                     #self.inst_label_train.extend(curr_inst_label)
#                 else:
#                     file_handler = open('./pickle/instsmel_test_pickled_{}.dat'.format(inst), 'rb+')
#                     curr_inst_spec = pickle.load(file_handler)
#                     file_handler.close()
#                     self.inst_spec_test.extend(curr_inst_spec)
#                     #self.inst_label_test.extend(curr_inst_label)

#     def __len__(self):
#         if(not self.test):
#             return len(self.inst_spec_train)
#         else:
#             return len(self.inst_spec_test)

#     def __getitem__(self, idx):
#         if(not self.test):
#             return self.inst_spec_train[idx]
#         else:
#             return self.inst_spec_test[idx]

class InstrumentsDatasetVAE(Dataset):
    def __init__(self, src='/home/nevronas/dataset/IRMAS-TrainingData', load=False, test=False):
        super(InstrumentsDatasetVAE, self).__init__()
        
        self.inst_spec_train = []
        self.inst_spec_test = []
        self.test = test

        insts = [
                'cel', # Cello
                'cla', # Clarinet
                'flu', # Flute
                'gac', # Acoustic guitar
                'gel', # Electric guitar
                'org', # Organ
                'pia', # Piano
                'sax', # Saxophone
                'tru', # Trumpet
                'vio'  # Violin
            ]

        if(load):
            for inst in insts:
                curr_inst_spec_train = []
                curr_inst_spec_test = []

                files = os.listdir(os.path.join(src, inst))
                i = 0
                for f in files:
                    print(f)
                    signal, fs = librosa.load(os.path.join(os.path.join(src, inst), f))
                    S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=40, n_fft=1024)
                    print(S.shape)
                    if(S.shape[1] < 19):
                        print("Chuck this sample!")
                        continue
                    S_array = []
                    for seg in range(0, S.shape[1] - 19, 19):
                        S_array.append(np.expand_dims(np.transpose(S[:, seg:seg + 19]), 0))
                    if(i < int(0.9*len(files))):
                        for seg in S_array:
                            print(seg.shape)
                            curr_inst_spec_train.append(seg)
                    else:
                        for seg in S_array:
                            curr_inst_spec_test.append(seg)
                    i = i+1
                file_handler = open('./pickle2/instsvae_train_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_train), file_handler)
                file_handler.close()
                file_handler = open('./pickle2/instsvae_test_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_test), file_handler)
                file_handler.close()

        else:

            for inst in insts:
                if(not self.test):
                    file_handler = open('./pickle2/instsvae_train_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_train.extend(curr_inst_spec)
                    
                else:
                    file_handler = open('./pickle2/insts_test_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_test.extend(curr_inst_spec)

    def __len__(self):
        if(not self.test):
            return len(self.inst_spec_train)
        else:
            return len(self.inst_spec_test)

    def __getitem__(self, idx):
        if(not self.test):
            return self.inst_spec_train[idx]
        else:
            return self.inst_spec_test[idx]
        
if __name__ == '__main__':
    #dataset = InstrumentsDataset(src='/home/nevronas/dataset/IRMAS-TrainingData', load=True)
    dataset = InstrumentsDatasetVAE(src='/home/nevronas/dataset/IRMAS-TrainingData', load=True)
