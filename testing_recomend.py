import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
# from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics





import torchmetrics
from copy import deepcopy
from pytorch_lightning import LightningModule
from torch import Tensor, FloatTensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple
from tqdm import tqdm



from pytorch_lightning.loggers import WandbLogger



import torch.nn.functional as tof


import torchmetrics.functional as mf



from collections import OrderedDict
import numpy as np

import itertools

import wandb

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from typing import Tuple, List

from clmr.datasets import get_dataset

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset



class TestDataset(Dataset):
    def __init__(self, dataset: Dataset, input_shape: List[int], transform: Compose):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape
        self.ignore_idx = []

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if idx in self.ignore_idx:
#             print(idx, self, 'mmmmmm')
            return self[idx + 1]

        audio, label, name = self.dataset[idx]
        
#         audio = audio[0]
        

        
        
        if audio.shape[1] < self.input_shape[1]:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        
        

        
        batch = torch.split(audio, self.input_shape[1], dim=1)
        
        batch = torch.cat(batch[:-1])
        
        batch = batch.unsqueeze(dim=1)
        
        
        if self.transform:
            batch = self.transform(batch)
        return batch, label, name

    def __len__(self) -> int:
        return len(self.dataset)



args = argparse.Namespace(accelerator=None, accumulate_grad_batches=None, amp_backend='native', amp_level=None,
                          audio_length=59049, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, batch_size=48,
                          benchmark=None, check_val_every_n_epoch=1, checkpoint_callback=None,
                          checkpoint_path='../../input/mine-checkpoints/encoder_1536_6148.ckpt',
                          classifier_head_checkpoint_path='../../input/mine-checkpoints/finetuner_with18gb_78_711.ckpt',
                          dataset='recomend', dataset_dir='./tracks/Intact conv', default_root_dir=None, detect_anomaly=False,
                          deterministic=None, devices=None, enable_checkpointing=True, enable_model_summary=True, enable_progress_bar=True,
                          fast_dev_run=False, finetuner_batch_size=256, finetuner_checkpoint_path='', finetuner_learning_rate=0.001,
                          finetuner_max_epochs=200, finetuner_mlp=0, flush_logs_every_n_steps=None, gpus=1, gradient_clip_algorithm=None,
                          gradient_clip_val=None, ipus=None, learning_rate=0.0003, limit_predict_batches=None, limit_test_batches=None,
                          limit_train_batches=None, limit_val_batches=None, log_every_n_steps=50, log_gpu_memory=None, logger=True,
                          max_epochs=None, max_steps=-1, max_time=None, min_epochs=None, min_steps=None, move_metrics_to_cpu=False,
                          multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=None, num_sanity_val_steps=2,
                          optimizer='Adam', overfit_batches=0.0, plugins=None, precision=32, prepare_data_per_node=None,
                          process_position=0, profiler=None, progress_bar_refresh_rate=None, projection_dim=64,
                          reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, sample_rate=22050,
                          seed=42, stochastic_weight_avg=False, strategy=None, supervised=0, sync_batchnorm=False, temperature=0.5,
                          terminate_on_nan=None, tpu_cores=None, track_grad_norm=-1, transforms_delay=0.3, transforms_filters=0.8,
                          transforms_gain=0.3, transforms_noise=0.01, transforms_pitch=0.6, transforms_polarity=0.8, transforms_reverb=0.6,
                          val_check_interval=None, weight_decay=1e-06, weights_save_path=None, weights_summary='top', workers=2)



class RegressionRecomend(pl.LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, classes_count: int, embed_hidden_dim:int = 128):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        
        
        
        
        
        
#         self.classifier_head = nn.Sequential(nn.Linear(self.hidden_dim, classes_count))
        
        
        
#         self.audio_linear = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         self.classes_layer = nn.Sequential(
#                 nn.Linear(classes_count, classes_count),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         print(self.hidden_dim+classes_count)
        
        
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim+classes_count, embed_hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1), #2
#             )


        self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, embed_hidden_dim),
#                 nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
#                 nn.Tanh(),
#                 nn.Dropout(0.5),
                 #0.5 0.1
                nn.Linear(embed_hidden_dim, 1), #2
            ) #canon-mlp
        
        
        #nn.Sequential(nn.Linear(self.hidden_dim+classes_count, 1))
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(embed_hidden_dim, embed_hidden_dim),
#                 nn.ReLU(),
# #                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1)#2
#             )
        
        
        
        
        
        #попробуем линейный слой чтобы не подгонять трансформер
        
        
#         self.map_for_encoder = nn.Sequential(
#             nn.Linear(in_features=self.hidden_dim, out_features=embed_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=embed_hidden_dim, out_features=embed_hidden_dim)
# #             nn.LayerNorm(embed_hidden_dim)
#         )
        
        
        
        
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_hidden_dim, nhead=2) #пока 2 было 2, 3,7 
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        
        
        
        
        #мапить каждую модальность в n vector с батч нормализацией
        
#         self.project_t = nn.Sequential()
#         self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
#         self.project_t.add_module('project_t_activation', self.activation)
#         self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        
        
        
        self.criterion = self.configure_criterion()

        self.cosine = torchmetrics.CosineSimilarity()
        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        preds = self._forward_representations(x, y) 
        
        #print(torch.sigmoid(preds), "PREDS", "\n"*2)
        
        loss = self.criterion(preds, y)
        return loss, preds
    
    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
                
#         with torch.no_grad():
#             classes = self.classifier_head(h0)
        
#         classes = torch.sigmoid(classes)

#         h = torch.cat([h0,classes],dim=1) #-1)





#         embeded = self.map_for_encoder(h0)[:, None, :]

#         h = self.transformer_encoder(embeded).squeeze()

        return torch.sigmoid(self.model(h0)) #tof.softmax(self.model(combined), dim=1) #torch.sigmoid(self.model(combined)) #(self.model(combined) > 0.5).float() * 1 #self.model(combined)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         y_bool = y > 0.5
        
#         _, preds = torch.max(preds, 1)
        
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()
        
    
    
    
    
    
    
    
    
#         preds = torch.sigmoid(preds)
        preds = (preds.cpu() > 0.5).int()
        y = y.cpu().int()   #.squeeze()
        
        
        
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
#         print(preds, y)
        self.log("Train/cosine", self.cosine(preds, y))
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/r2_score", mf.r2_score(preds, y))
        self.log("Train/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Train/precision", mf.precision(preds, y, average = "macro", num_classes=2))
        
        self.log("Train/ACC", metrics.accuracy_score(y, preds))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         _, preds = torch.max(preds, 1)
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()









#         preds = torch.sigmoid(preds)
        preds = (preds.cpu() > 0.5).int()
        y = y.cpu().int()    #.squeeze()
        
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
        
#         print(preds, y)
        
        
#         print(preds, y, "\n\nVAL!!!!!!!!!!!!!!!!!!!!!!\n\n")
        self.log("Valid/cosine", self.cosine(preds, y))
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/r2_score", mf.r2_score(preds, y))
        self.log("Valid/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Valid/precision", mf.precision(preds, y, average = "macro", num_classes=2))
        
        self.log("Valid/ACC", metrics.accuracy_score(y, preds))
        
        
        
        
        
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        criterion = torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#!! pos_weight=FloatTensor([1.66])!!! #nn.CrossEntropyLoss() #nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.finetuner_learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.00001,#0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
    
    def extract_representations(self, dataloader: DataLoader) -> Dataset:

        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                h0 = self.encoder(x)
                representations.append(h0)
                ys.append(y)

        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset
    



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="SimCLR")
#     parser = Trainer.add_argparse_args(parser)

#     config = yaml_config_hook("./config/config.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))
    
#     parser.add_argument(f"--classifier_head_checkpoint_path",default="", type=type("a"))
    
    
    
#     args = parser.parse_args()

# print(args, "ARGS")



pl.seed_everything(args.seed)
args.accelerator = None





# ------------
# encoder
# ------------
if not os.path.exists(args.checkpoint_path):
    raise FileNotFoundError("That checkpoint does not exist")

train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

# ------------
# dataloaders
#     ------------



train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="full")
valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="full")
test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

#print(train_dataset.fl)




encoder = SampleCNN(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=args.supervised,
    out_dim=test_dataset.n_classes,
)

n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
encoder.load_state_dict(state_dict)


# encoder.eval()


cl = ContrastiveLearning(args, encoder)
cl.eval()
cl.freeze()



module = RegressionRecomend(
    args,
    encoder,
    hidden_dim = n_features,
    classes_count = test_dataset.n_classes
)




contrastive_train_dataset = ContrastiveDataset(
    train_dataset,
    input_shape=(1, args.audio_length),
    transform=Compose(train_transform),
)

contrastive_valid_dataset = ContrastiveDataset(
    valid_dataset,
    input_shape=(1, args.audio_length),
    transform=Compose(train_transform),
)

contrastive_test_dataset = TestDataset(
    test_dataset,
    input_shape=(1, args.audio_length),
    transform=None,
)

train_loader = DataLoader(
    contrastive_train_dataset,
    batch_size=32,#args.finetuner_batch_size,
    num_workers=args.workers,
    shuffle=True,
)

valid_loader = DataLoader(
    contrastive_valid_dataset,
    batch_size=32,#args.finetuner_batch_size,
    num_workers=args.workers,
    shuffle=True,
)


train_representations_dataset = module.extract_representations(train_loader)
train_loader = DataLoader(
    train_representations_dataset,
    batch_size=32,#args.batch_size,
    num_workers=args.workers,
    shuffle=True,
)

valid_representations_dataset = module.extract_representations(valid_loader)
valid_loader = DataLoader(
    valid_representations_dataset,
    batch_size=32,#args.batch_size,
    num_workers=args.workers,
    shuffle=False,
)



class RegressionRecomend(pl.LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, classes_count: int, embed_hidden_dim:int = 128, alpha:float = 1):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        
        
        
        
        
        
        self.classifier_head = nn.Sequential(nn.Linear(self.hidden_dim, classes_count))
        
        
        
        
        
        
#         self.audio_linear = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         self.classes_layer = nn.Sequential(
#                 nn.Linear(classes_count, classes_count),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         print(self.hidden_dim+classes_count)
        
        
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim+classes_count, embed_hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1), #2
#             )

#---------------------------------------------canon-mlp-------------------------
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim, embed_hidden_dim, bias=False),
# #                 nn.LayerNorm(self.hidden_dim),
#                 nn.ReLU(),
# #                 nn.Tanh(),
# #                  nn.Dropout(0.1),
#                  #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1, bias=False), #2
#             ) #canon-mlp
#---------------------------------------------canon-mlp-------------------------
        
#!------------------double loss----------------------        
    
#         self.classification_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim//2, bias = False),
#             nn.ReLU(),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim//2, classes_count, bias = False)
#         )
        
#         self.like_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = False),
#             nn.ReLU(),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, 1, bias = False)
#         )
         
#         self.alpha = alpha
        
        
        
#!------------------double loss----------------------        
        
        
        #---------------------------------------------NOT LATE FUSION-----------------------------------------------
        
        
#         self.table_model = nn.Sequential(
#             nn.Linear(classes_count, classes_count//2),
#             nn.ReLU(),
#             nn.Linear(classes_count//2, classes_count//2)
#         )
        
#         self.audio_model = nn.Sequential(
#             nn.Linear(self.hidden_dim, embed_hidden_dim),
# #                 nn.LayerNorm(self.hidden_dim),
#             nn.ReLU(),
#             #nn.Dropout(0.1),
#             nn.Linear(embed_hidden_dim, embed_hidden_dim)
#         )
        
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim+classes_count, embed_hidden_dim),
#                 nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_hidden_dim, 1)
        )

        #---------------------------------------------NOT LATE FUSION-----------------------------------------------
        
        
        
        #nn.Sequential(nn.Linear(self.hidden_dim+classes_count, 1))
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim+classes_count//2, embed_hidden_dim),
#                 nn.ReLU(),
# #                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1)#2
#             )
        
        
        
        
        
        #попробуем линейный слой чтобы не подгонять трансформер
        
        
#         self.map_for_encoder = nn.Sequential(
#             nn.Linear(in_features=self.hidden_dim, out_features=embed_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=embed_hidden_dim, out_features=embed_hidden_dim)
# #             nn.LayerNorm(embed_hidden_dim)
#         )
        
        
        
#         print(self.hidden_dim+classes_count//2)
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim+classes_count, nhead=3) #пока 2 было !2! , 3, !5!, 7, 23
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        
        
        
        
        #мапить каждую модальность в n vector с батч нормализацией
        
#         self.project_t = nn.Sequential()
#         self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
#         self.project_t.add_module('project_t_activation', self.activation)
#         self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        
        
        
#         self.criterions = self.configure_criterions()
        
        
        self.criterion = self.configure_criterion()
        
        
        self.cosine = torchmetrics.CosineSimilarity()
        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
#         likebles, classes_pred, classes_true = self._forward_representations(x, y) 
        preds = self._forward_representations(x, y) 
        
#         loss = self.calculate_double_loss(likebles, y, classes_pred, classes_true)
        
        #print(torch.sigmoid(preds), "PREDS", "\n"*2)
        
        loss = self.criterion(preds, y)

        return loss, preds #preds#likebles
    
    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
                
                
#!--------------------------------------------------------------------                 
#         with torch.no_grad():
#             classes = self.classifier_head(h0)
        
#         classes = torch.sigmoid(classes)        
                
#         h = torch.cat([h0,classes],dim=1)        
#!--------------------------------------------------------------------                 
            
            
            
            
            
        #!--------------------------------------------------------------------        
        with torch.no_grad():
            classes = self.classifier_head(h0)
        
        classes = torch.sigmoid(classes)
        
        
#         table_coded = self.table_model(classes)
#         audio_coded = self.audio_model(h0)
        
        h = torch.cat([classes,h0],dim=1) #-1)
        #!--------------------------------------------------------------------




#         embeded = self.map_for_encoder(h0)[:, None, :]

#         h = self.transformer_encoder(h[:, None, :]).squeeze()

        return torch.sigmoid(self.model(h))
#torch.sigmoid(self.model(h0))

#torch.sigmoid(self.like_head(h0)), torch.sigmoid(self.classification_head(h0)), classes 

#tof.softmax(self.model(combined), dim=1) #torch.sigmoid(self.model(combined)) #(self.model(combined) > 0.5).float() * 1 #self.model(combined)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         y_bool = y > 0.5
        
#         _, preds = torch.max(preds, 1)
        
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()
        
    
    
    
    
    
    
    
    
#         preds = torch.sigmoid(preds)



        #----------------------------------------BCE----------------------------------------------------------
#         preds = (preds.cpu() > 0.5).int()
#         y = y.cpu().int()    #.squeeze()
        #----------------------------------------BCE----------------------------------------------------------
        
        
        
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
#         print(preds, y)
        self.log("Train/cosine", self.cosine(preds, y))
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/r2_score", mf.r2_score(preds, y))
        self.log("Train/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Train/precision", mf.precision(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2, multiclass=True))
        
#         self.log("Train/ACC", metrics.accuracy_score(y, preds))
        
    
    
    
        self.log("Train/recall", mf.recall(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2,multiclass=True))
        
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         _, preds = torch.max(preds, 1)
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()









#         preds = torch.sigmoid(preds)




        #----------------------------------------BCE----------------------------------------------------------
#         preds = (preds.cpu() > 0.5).int()
#         y = y.cpu().int()    #.squeeze()
        #----------------------------------------BCE----------------------------------------------------------
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
        
#         print(preds, y)
        
        
#         print(preds, y, "\n\nVAL!!!!!!!!!!!!!!!!!!!!!!\n\n")
        self.log("Valid/cosine", self.cosine(preds, y))
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/r2_score", mf.r2_score(preds, y))
        self.log("Valid/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Valid/precision", mf.precision(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2, multiclass=True))
        
#         self.log("Valid/ACC", metrics.accuracy_score(y, preds))
        
    
    
    
        self.log("Valid/recall", mf.recall(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2,multiclass=True))
        
        
        
        
        self.log("Valid/loss", loss)
        return loss

    def configure_criterions(self) -> nn.Module:
        criterion_1 = nn.BCELoss() #torch.nn.MSELoss() #torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#!! pos_weight=FloatTensor([1.66])!!! #nn.CrossEntropyLoss() #nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        criterion_2 = nn.BCELoss()
        return (criterion_1, criterion_2)

    def configure_criterion(self) -> nn.Module:
        criterion = nn.BCELoss() #torch.nn.MSELoss() #torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#!! pos_weight=FloatTensor([1.66])!!! #nn.CrossEntropyLoss() #nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        return criterion


    def calculate_double_loss(self,likebles_pred, likebles_true, classes_pred, classes_true):
#         print(classes_pred.shape, classes_true.shape)
        classes_loss = self.criterions[0](classes_pred, classes_true)
        binary_loss = self.criterions[1](likebles_pred, likebles_true)
        
        
        
#         self.log("classes_pred", classes_pred)
#         self.log("classes_true", classes_true)
        
        
        self.log("binary_loss", binary_loss)
        self.log("classes_loss", classes_loss)
        
        return binary_loss + self.alpha*classes_loss

    def configure_optimizers(self) -> dict:
        
        
        
        #!!!!!!!!!!!!В ТРАНСФОРМЕРЕ И ЭНКОДЕРАХ ОПТИМИЗИРОВАЛСЯ ТОЛЬКО ЛИНЕЙНЫЙ ВЫХОДНОЙ СЛОЙ model!!!!!!! ВСЁ ПЕРЕДЕЛАТЬ!!!!!!
        
        #self.table_model.parameters(), self.audio_model.parameters(), self.model.parameters()
        #self.table_model.parameters(), self.model.parameters(), self.transformer_encoder.parameters()
        
        all_params = itertools.chain(self.model.parameters())
        
        
        optimizer = torch.optim.Adam(
            all_params,#all_params,#self.model.parameters(),
            lr=self.hparams.finetuner_learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.00001,#0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
    
    def extract_representations(self, dataloader: DataLoader) -> Dataset:

        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                h0 = self.encoder(x)
                representations.append(h0)
                ys.append(y)

        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset
    



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="SimCLR")
#     parser = Trainer.add_argparse_args(parser)

#     config = yaml_config_hook("./config/config.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))
    
#     parser.add_argument(f"--classifier_head_checkpoint_path",default="", type=type("a"))
    
    
    
#     args = parser.parse_args()

# print(args, "ARGS")



pl.seed_everything(args.seed)
args.accelerator = None


args.workers = 2


# ------------
# encoder
# ------------
encoder = SampleCNN(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=args.supervised,
    out_dim=test_dataset.n_classes,
)

n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
encoder.load_state_dict(state_dict)


# encoder.eval()


cl = ContrastiveLearning(args, encoder)
cl.eval()
cl.freeze()



module = RegressionRecomend(
    args,
    encoder,
    hidden_dim = n_features,
    classes_count = test_dataset.n_classes
)

state_dict = load_finetuner_checkpoint(args.classifier_head_checkpoint_path)
module.classifier_head.load_state_dict(state_dict)


module.classifier_head.eval()
module.classifier_head.requires_grad_(False)

module.encoder.requires_grad_(False)
module.encoder.eval()


print(module.model)



# print(module.like_head, module.classification_head)




#     whole_model: nn.Module,
#     test_dataset: Dataset,
#     dataset_name: str,
#     audio_length: int,
#     device
if args.finetuner_checkpoint_path:
    state_dict = OrderedDict({
        k:v
        for k,v in torch.load(args.finetuner_checkpoint_path, map_location=torch.device("cuda:0"))["state_dict"].items()
    }) #torch.load(args.finetuner_checkpoint_path, map_location=torch.device("cpu")) #load_finetuner_checkpoint(args.finetuner_checkpoint_path)
#         module.model.load_state_dict(state_dict)
    module.load_state_dict(state_dict)

#         print(module.state_dict())


#         state_dict = torch.load(args.checkpoint_path, map_location="cuda:0")
#         module.model.load_state_dict(state_dict)

    early_stop_callback = EarlyStopping(
        monitor="Valid/loss", patience=20, verbose=False, mode="min"
    ) #40
    print("INIT TRAININ")
    trainer = Trainer.from_argparse_args(
        args,
        logger=WandbLogger(project="recomend_finetune"),
        #TensorBoardLogger(
            #"runs", name="CLMRv2-eval-{}".format(args.dataset)
        #),
        max_epochs=-1,#args.finetuner_max_epochs,
        callbacks=[early_stop_callback],
        accelerator='gpu', 
        log_every_n_steps=1,
#             check_val_every_n_epoch=5

    )
    print("START TRAININ")
    trainer.fit(module, train_loader, valid_loader)


else:
    early_stop_callback = EarlyStopping(
        monitor="Valid/loss", patience=20, verbose=False, mode="min"
    ) #40 20
    print("INIT TRAININ")
    trainer = Trainer.from_argparse_args(
        args,
        logger=WandbLogger(project="recomend_finetune"),
        #TensorBoardLogger(
            #"runs", name="CLMRv2-eval-{}".format(args.dataset)
        #),
        max_epochs=-1,#args.finetuner_max_epochs,
        callbacks=[early_stop_callback],
        accelerator='gpu', 
        log_every_n_steps=1,
#             check_val_every_n_epoch=5

    )
    print("START TRAININ")
    trainer.fit(module, train_loader, valid_loader)

print("AT EVAL")





# module.eval()
# module.freeze()

# wandb.finish()
#  def concat_clip(audio, audio_length: float) -> Tensor:
#         batch = torch.split(audio, audio_length, dim=1)
#         batch = torch.cat(batch[:-1])
#         batch = batch.unsqueeze(dim=1)
#         return batch


def evaluate(
    whole_model: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device
) -> dict:
    est_array = []
    gt_array = []
    names = []
    
    whole_model = whole_model.to(device)
    whole_model.eval()
    whole_model.freeze()
    whole_model.encoder.eval()
#     whole_model.classifier_head.eval()
    
    
    
    
    average_precision = torchmetrics.AveragePrecision(pos_label=1)
#     if finetuned_head is not None:
#         finetuned_head = finetuned_head.to(device)
#         finetuned_head.eval()
    
    with torch.no_grad():
#         for idx in tqdm(range(len(test_dataset))):
#             _, label = test_dataset[idx]
#             batch = test_dataset.concat_clip(idx, audio_length)
#             batch = batch.to(device)

    
        
        for batch, label, name in tqdm(test_dataset):        
            batch = batch.to(device)

            
#             print(batch.shape)
#             h0 = whole_model.encoder(batch)
#             print(h0.shape)
            
            h0 = []
#             for cutted in batch:    
#                 h0.append(whole_model.encoder(cutted[None]))
            
            ld = DataLoader(batch, batch_size=16)
            for cutted in ld:
                for out in whole_model.encoder(cutted):
                    h0.append(out)
            
            
            
            h0 = torch.stack(h0)
            h0 = torch.squeeze(h0, 1)
            
            
            
            
            
            
#!--------------------------------------------------------------------!            
#             classes = whole_model.classifier_head(h0)
        
#             classes = torch.sigmoid(classes)
            
#             h = torch.cat([h0,classes],dim=1)
#!--------------------------------------------------------------------!        
            
    
    
#!--------------------------------------------------------------------!
            classes = whole_model.classifier_head(h0)
        
            classes = torch.sigmoid(classes)


#             table_coded = whole_model.table_model(classes)
#             audio_coded = whole_model.audio_model(h0)

            h = torch.cat([classes,h0],dim=1) #-1)
            
#!--------------------------------------------------------------------!           
            
#             h = whole_model.transformer_encoder(h[:, None, :]).squeeze()
            
##             print(h0.shape)
            
    
            #classes = torch.sigmoid(whole_model.classifier_head(h0))
            
##            print(classes.shape)

#             h = whole_model.transformer_encoder(whole_model.map_for_encoder(torch.cat([h0,classes],dim=1))[:, None, :]).squeeze()
            
    
            #h = torch.cat([h0,classes],dim=1)
    
    
#             output = torch.sigmoid(whole_model.model(torch.cat([h0,torch.sigmoid(whole_model.classifier_head(h0))],dim=1)))#-1))
            
##             print(output.shape)

#             output = torch.sigmoid(output)
    
##             print(output.shape)
            
            
        
#             output = torch.sigmoid(whole_model.model(h0))
            
    
    
#             h = whole_model.transformer_encoder(whole_model.map_for_encoder(h0)[:, None, :]).squeeze()
            
#             track_prediction = whole_model.model(h).mean(dim=0)
    
            track_prediction = torch.sigmoid(whole_model.model(h)).mean(dim=0) #output.mean(dim=0)
            
##             print(track_prediction,"OUTPUT")
            
            
            est_array.append(track_prediction)
            gt_array.append(label)
            names.append(name)

    est_array = torch.stack(est_array, dim=0).cpu()
    
    gt_array = torch.stack(gt_array, dim=0).cpu().int()    #.squeeze()#.numpy()
    
#     print(est_array, gt_array, "PRED/TRUE")
    
#     _, est_array = torch.max(est_array, 1)

   
    #--------------------------------BCE-------------------------------
#     est_array = (est_array > 0.5).int()
    #--------------------------------BCE-------------------------------
    
#     print("PRED")
#     print(est_array)
    
#     print("TRUE")
#     print(gt_array)
    
#     print("NAMES")
#     print(names)
    
    
    
    for name, pred, true in zip(names, est_array, gt_array):
        print(f"Name: {name}, Prediction: {pred}, True: {true}")
    
    
    
    r2score = mf.r2_score(est_array, gt_array)
    msle = mf.mean_squared_log_error(est_array, gt_array)
#     accuracy = metrics.accuracy_score(gt_array, est_array)
    ap = average_precision(est_array, gt_array)
    
    pr = mf.precision(est_array, gt_array, average = "macro", num_classes=2,multiclass=True)
    rc = mf.recall(est_array, gt_array, average = "macro", num_classes=2,multiclass=True)
    
    
    
    
    
    
#     print(r2score, msle)
    
    
#     print(cosin)
    
    
    return {
        "msle": msle,
        "r2score": r2score,
#         "accuracy":accuracy,
        "average_precision":ap,
        "precision":pr,
        "recall":rc
    }
  
  
  


test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

# for a, b, c in test_dataset:
#     print(a,b,c)


contrastive_test_dataset = TestDataset(
    test_dataset,
    input_shape=(1, args.audio_length),
    transform=None,
)


device = "cuda:0" if args.gpus else "cpu"
results = evaluate(
    module,
    contrastive_test_dataset,
    args.dataset,
    args.audio_length,
    device=device
)
wandb.log(results)
wandb.finish()
print(results)
