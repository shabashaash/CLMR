import torchaudio
from torch import Tensor, FloatTensor, IntTensor
from typing import Tuple
from glob import glob
from clmr.datasets import Dataset
import random
import os
import fpl_reader



class PLAYLISTS(Dataset):
    """Create a Dataset for any folder of playlist files.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        src_ext_audio (str): The extension of the audio files to analyze.
    """
    
    
    
    st_labels = {'More Rhythmic Calmness': 0,
 'Vocal': 1,
 'Great Battle': 2,
 'Pop': 3,
 'BG': 4,
 'Go POP': 5,
 'Im Gonna Cry': 6,
 'Rhuthmic Calmness': 7,
 'Just Battle': 8,
 'Chill Beats': 9,
 'Peace': 10,
 'Elektro': 11,
 'Hope': 12,
 'Sporadic': 13,
 'Comfort of existance': 14,
 'Great hope': 15,
 'Uplifting': 16,
 'Grand Despair': 17,
 'Choir': 18,
 'Ambient': 19,
 'SUSpense': 20,
 'Grand Calmness': 21,
 'Great Classic': 22,
 'Uncertainty': 23,
 'Sewerslvt': 24,
 'Jazzy': 25,
 'Ambient Nothingness': 26,
 'Mysterious': 27,
 'BG2': 28,
 'ADHD': 29,
 'Sad': 30,
 'The': 31,
 'That moment': 32,
 'Emotional Movement': 33}


    def __init__(
        self,
        root,
        playlist_paths: str = "../../input/textplaylists/Converted/*",
        src_ext_audio: str = ".wav",
        n_classes: int = 1,
        subset:str = "train"
    ) -> None:  
        
        super(PLAYLISTS, self).__init__(root)
#         self._path = root #"../../input/trackswav/converted/"#root



        self._src_ext_audio = src_ext_audio
        self._playlist_paths = glob(playlist_paths) #glob("../../input/playlists-fpl/Converted/*")#glob(root+'/'+playlist_paths+'/*') #glob("../../input/playlists/Converted/*")#glob(root+'/'+playlist_paths+'/*')
        
        
        self.true_paths = set(glob(root+"/*.wav"))

        self.data = {}
        
        self.subset = subset
        
        
        train_valid_split = 0.9
        
        test_store = {}
        
        
        
        print(len(self.true_paths))
        
        for i in glob("../../input/trackswav/converted/*.wav"):  #8gb
            self.true_paths.add(i)
        
        print(len(self.true_paths))
        
        
        
        
        
        tracks = [i.split('/')[-1].split('-converted')[0] for i in self.true_paths]

        tmp_paths = []

        for name in tracks:
            words_local = []
            for word in name.split(' '):

                tmp_ = ''.join(e for e in word if e.isalnum())
                
                if 'umib' not in tmp_:
                    tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                tmp_ = ' '.join(tmp_.split())

                words_local.append(tmp_)

            tmp_paths.append(' '.join([i for i in words_local if len(i)>0]))
        
#         tracks = tmp_paths
        
        
        
        data_dict = dict(zip(tmp_paths, self.true_paths))
        
        
        print(len(data_dict), "FolderPaths")
        
#         for i in list(data_dict.values()):
#             if 'voiceless' in i:
#                 print(i)
        
        
        
        for playlist in self._playlist_paths:
            with open(playlist, encoding = "utf-8") as f:
                for line in f.read().split(';'):
                    if len(line)>0:


                        words_local = []
                        for word in line.split(' '):

                            tmp_ = ''.join(e for e in word if e.isalnum())
                            
                            
                            if 'umib' not in tmp_:
                                tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                            tmp_ = ' '.join(tmp_.split())

                            words_local.append(tmp_)

                        line_ = ' '.join([i for i in words_local if len(i)>0])
                        if line_ in data_dict.keys():
#                             print(line_, 'S_in_TRUE')
                            if data_dict[line_] in self.data:
                                self.data[data_dict[line_]].add(playlist.split('/')[-1][:-4])
                            else:
                                self.data[data_dict[line_]] = set()
                                self.data[data_dict[line_]].add(playlist.split('/')[-1][:-4])

        
#         for playlist in self._playlist_paths:
#             with open(playlist, 'rb') as handle:
#                 print(playlist,"playlist")
                
#                 playlist_c = str(fpl_reader.read_playlist(handle.read())).split('(\'file_name\',')[1:]
#                 for line in playlist_c:
                    
#                     filename = line.split('),')[0].split('\\')[-1][:-5]
                    
                    
                    
                    
#                     words = filename.split()
                    
# #                     max([[len(words[i]),i] for i in range(len(words))])
                    
                    
                    
#                     s_ = self._path+filename+'-converted.wav'        
#                     if s_ in self.true_paths:
#                         print(s_, 'S_in_TRUE')
#                         if s_ in self.data:
#                             self.data[s_].add(playlist.split('/')[-1][:-4])
#                         else:
#                             self.data[s_] = set()

        self.data = list(self.data.items())
        self.n_classes = len(self.st_labels)
        
        
        if subset == "train":
            self.data = self.data[:int(train_valid_split*len(self.data))]
        if subset == "valid":
            self.data = self.data[int(train_valid_split*len(self.data))+1:]
        if subset=='test':       
            self.data = random.sample(self.data, 100) #17 66
            print(self.data,'TESTFINALE')

            
            
        print(subset, len(self.data), self.st_labels, len(self.st_labels))
        if len(self.data) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    root
                )
            )

    def file_path(self, n: int) -> str:
        fp = self.data[n][0]
        return fp

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple [Tensor, Tensor]: ``(waveform, label)``
        """
        
        audio, _ = self.load(n)
        label_bin = [0 for _ in range(self.n_classes)]
        labels = self.data[n][1]
        for label in labels:
            label_bin[self.st_labels[label]] = 1
            
#         if self.subset == "test":
#             return audio, FloatTensor(label_bin), self.data[n][0]
        
        return audio, FloatTensor(label_bin), self.data[n][0]

    def __len__(self) -> int:
        return len(self.data)
