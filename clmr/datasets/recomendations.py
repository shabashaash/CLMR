import os
from glob import glob
from torch import Tensor, FloatTensor, LongTensor
from typing import Tuple

import os
from clmr.datasets import Dataset
import random

class RECOMENDATIONS(Dataset):
    """Create a Dataset for any folder of audio files.
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
        root: str,
        subset: str = "full",
        src_ext_audio: str = ".wav",
        n_classes: int = 1,#len(self.st_labels),
        t_mode: bool = True, #в плейлисте положительные или негативные примеры
        playlist_path: str = "../../input/truelabelstext/Default.txt"
        
    ) -> None:
        super(RECOMENDATIONS, self).__init__(root)

        self._path = root
        self._src_ext_audio = src_ext_audio
        self.n_classes = len(self.st_labels)#25#n_classes
        
        self.t_mode = t_mode
        
        true_paths = set(glob(root+"/*.wav"))

        for i in glob("../../input/trackswav/converted/*.wav"):  #8gb
            true_paths.add(i)
        
        
        
        true_paths = list(true_paths)
        
        
        
        tracks = [i.split('/')[-1].split('-converted')[0] for i in true_paths]

        tmp_paths = []
        full_paths = []
        
        
        
        
        
        for j in range(len(tracks)):
            words_local = []
            for word in tracks[j].split(' '):

                tmp_ = ''.join(e for e in word if e.isalnum())
                
                if 'umib' not in tmp_:
                    tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                tmp_ = ' '.join(tmp_.split())

                words_local.append(tmp_)
            
            tmp_ = ' '.join([i for i in words_local if len(i)>0])

            if len(tmp_) > 0 and tmp_ not in tmp_paths:
                tmp_paths.append(tmp_)
                full_paths.append(true_paths[j])

        
        self.playlist_info = []
        for path in open(playlist_path, encoding="utf-8").read().split(';'):
            name = os.path.basename(path).split('.')[0]
            words_local = []
            for word in name.split(' '):

                tmp_ = ''.join(e for e in word if e.isalnum())
                
                if 'umib' not in tmp_:
                    tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                tmp_ = ' '.join(tmp_.split())

                words_local.append(tmp_)
            
            tmp_ = ' '.join([i for i in words_local if len(i)>0])
            
            if len(tmp_) > 0:
                self.playlist_info.append(tmp_)

        if subset == "test":
            self.fl = dict(random.sample(dict(zip(tmp_paths, full_paths)).items(), 100))
            print(self.fl, "LOOK AT MEE")
        else:
            self.fl = {}
            pos = {}
            neg = {}
            
            for i in range(len(tmp_paths)):
                label = self.t_mode == (tmp_paths[i] in self.playlist_info)
                if label == 1:
                    pos[tmp_paths[i]] = full_paths[i]
                else:
                    neg[tmp_paths[i]] = full_paths[i]
                

            
            print(len(pos), "pos_len")
            print(len(neg), "neg_len")
            
            
            
            
            neg = dict(random.sample(neg.items(), min(len(pos), len(neg))))

            
#             print(pos, neg, "pos,neg")
            
            self.fl = neg
            self.fl.update(pos)

            print(len(self.fl), "all_len")
            
        print(len(self.fl), len(self.playlist_info), self.playlist_info[0], self.n_classes)
        
        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )

    def file_path(self, n: int) -> str:
        return list(self.fl.values())[n]

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple [Tensor, Tensor]: ``(waveform, label)``
        """
        audio, _ = self.load(n)
        label = [1 if self.t_mode == (list(self.fl.keys())[n] in self.playlist_info) else 0]
        return audio, FloatTensor(label), list(self.fl.values())[n]

    def __len__(self) -> int:
        return len(self.fl)
