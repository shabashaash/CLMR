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
def evaluate(
    whole_model: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device
) -> dict:
    est_array = []
    gt_array = []

    whole_model = whole_model.to(device)
    whole_model.eval()
    whole_model.freeze()
    
    average_precision = torchmetrics.AveragePrecision(pos_label=1)
#     if finetuned_head is not None:
#         finetuned_head = finetuned_head.to(device)
#         finetuned_head.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            _, label = test_dataset[idx]
            batch = test_dataset.concat_clip(idx, audio_length)
            batch = batch.to(device)
            print(batch.shape)
            #h0 = whole_model.encoder(batch)
            h0 = whole_model.encoder(batch)
            
##             print(h0.shape)
            
    
            #classes = torch.sigmoid(whole_model.classifier_head(h0))
            
##            print(classes.shape)

##             h = whole_model.transformer_encoder(whole_model.map_for_encoder(torch.cat([h0,classes],dim=1))[:, None, :]).squeeze()
            
    
            #h = torch.cat([h0,classes],dim=1)
    
    
#             output = torch.sigmoid(whole_model.model(torch.cat([h0,torch.sigmoid(whole_model.classifier_head(h0))],dim=1)))#-1))
            
##             print(output.shape)

#             output = torch.sigmoid(output)
    
##             print(output.shape)
            
            
        
#             output = torch.sigmoid(whole_model.model(h0))
            
    
    
#             h = whole_model.transformer_encoder(whole_model.map_for_encoder(h0)[:, None, :]).squeeze()
            
#             track_prediction = whole_model.model(h).mean(dim=0)
    
            track_prediction = whole_model.model(h0).mean(dim=0) #output.mean(dim=0)
            
##             print(track_prediction,"OUTPUT")
            
            
            est_array.append(track_prediction)
            gt_array.append(label)


    est_array = torch.stack(est_array, dim=0).cpu()
    
    gt_array = torch.stack(gt_array, dim=0).cpu().int()    #.squeeze()#.numpy()
    
#     print(est_array, gt_array, "PRED/TRUE")
    
#     _, est_array = torch.max(est_array, 1)

   

    est_array = (est_array > 0.5).int()
    
    print(est_array, gt_array, "PRED/TRUE")
    
    
    r2score = mf.r2_score(est_array, gt_array)
    msle = mf.mean_squared_log_error(est_array, gt_array)
    accuracy = metrics.accuracy_score(gt_array, est_array)
    ap = average_precision(est_array, gt_array)
    
    pr = mf.precision(est_array, gt_array, average = "macro", num_classes=2)
    
    
    
#     print(r2score, msle)
    
    
#     print(cosin)
    
    
    return {
        "msle": msle,
        "r2score": r2score,
        "accuracy":accuracy,
        "average_precision":ap,
        "precision":pr
    }






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
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    parser.add_argument(f"--classifier_head_checkpoint_path",default="", type=type("a"))
    
    
    
    args = parser.parse_args()
    
#     print(args, "ARGS")
    
    
    
    pl.seed_everything(args.seed)
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

    # ------------
    # dataloaders
#     ------------



    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="full")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="full")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

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

    contrastive_test_dataset = ContrastiveDataset(
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

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()
    

    
    module = RegressionRecomend(
        args,
        encoder,
        hidden_dim = n_features,
        classes_count = test_dataset.n_classes
    )
    
#     state_dict = load_finetuner_checkpoint(args.classifier_head_checkpoint_path)
#     module.classifier_head.load_state_dict(state_dict)
    
    
#     module.classifier_head.eval()
#     module.classifier_head.requires_grad_(False)

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
    
    '''if args.finetuner_checkpoint_path:
        state_dict = OrderedDict({
         k:v
         for k,v in torch.load(args.finetuner_checkpoint_path, map_location=torch.device("cuda:0"))["state_dict"].items()
        })
        module.load_state_dict(state_dict)'''
    
    
    
    module.eval()
    module.freeze()
    
    
    device = "cuda:0" if args.gpus else "cpu"
    results = evaluate(
        module,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device=device
    )
    print(results)
