import os
from .dataset import Dataset
from .audio import AUDIO
from .librispeech import LIBRISPEECH
from .gtzan import GTZAN
from .magnatagatune import MAGNATAGATUNE
from .million_song_dataset import MillionSongDataset
from .playlists import PLAYLISTS
from .recomendations import RECOMENDATIONS


def get_dataset(dataset, dataset_dir, subset, download=True): #playlist_paths

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "audio":
        d = AUDIO(root=dataset_dir)
    elif dataset == "librispeech":
        d = LIBRISPEECH(root=dataset_dir, download=download, subset=subset)
    elif dataset == "gtzan":
        d = GTZAN(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    elif dataset == "msd":
        d = MillionSongDataset(root=dataset_dir, subset=subset)
    elif dataset == "playlists":
        d = PLAYLISTS(root=dataset_dir, subset=subset)
    elif dataset == "recomend":
        d = RECOMENDATIONS(root=dataset_dir, subset=subset)
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
