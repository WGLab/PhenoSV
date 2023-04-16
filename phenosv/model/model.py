import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import model.model_blocks as mb
import model.operation_function as of
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics




#reference:  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

class LocalSVModule(pl.LightningModule):
    def __init__(self, input_dim, input_middle_dim, model_dim, num_heads, number_layers_encoder,
                 attention_dim, classifier_hidden, lr, warmup, max_iters, dropout=0.0,
                 temperature=1,weight_decay=0.0001,scheduler='multistep',
                 teacher_model=None, beta=0.5,optimizer='adam',
                 indicator_type='sv', seq_pad = False, max_relative_position=0, pos_weight=None, clip_pad = True):

        """
        Inputs:
        input_dim - original dimensionality of the input (element feature dimension, 271)
        model_dim - Hidden dimensionality to use inside the Transformer (256)
        num_heads - Number of heads to use in the Multi-Head Attention blocks (4)
        number_layers_encoder - Number of encoder blocks to use. (16)
        attention_dim - dimensionality to use for aggregating element-wise embedding to SV embedding:
                        -1: use sentence embedding, 0: maxpooling, positive number: MIL
        classifier_hidden - dimensionality of final classifier hidden layer (256)
        lr - Learning rate in the optimizer
        warmup - Number of warmup steps. Usually between 50 and 500
        max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
        dropout - Dropout to apply inside the model (0.25)
        outside_sv - set true for global model
        indicator_type - when outside_sv is True, which indicator type to use for maxpooling, choose from gene and sv; if set to both, model will
        yeld two predictions: sv local pred and sv-gene pair pred
        """
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.pos_weight is not None:
            self.register_buffer('posweight',torch.tensor([self.hparams.pos_weight]))
            self.criterion1 = nn.BCEWithLogitsLoss(reduction='none',pos_weight=self.posweight)
        else:
            self.criterion1 = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.accuracy = torchmetrics.Accuracy()
        if self.hparams.attention_dim==-1:
            sentence_embedding=True
        else:
            sentence_embedding = False
        self.feature_extractor = mb.LocalFeatureExtractor(input_dim = self.hparams.input_dim,
                                                          input_middle_dim = self.hparams.input_middle_dim,
                                                          model_dim = self.hparams.model_dim,
                                                          num_heads = self.hparams.num_heads,
                                                          number_layers_encoder = self.hparams.number_layers_encoder,
                                                          dropout=self.hparams.dropout,
                                                          sentence_embedding=sentence_embedding,
                                                          max_relative_position = self.hparams.max_relative_position,
                                                          clip_pad = self.hparams.clip_pad)
        self.classifier = mb.ClassificationLayer(input_dim=self.hparams.model_dim,
                                                 middel_dim=self.hparams.classifier_hidden,
                                                 attention_dim=self.hparams.attention_dim)
        self.teacher_model = self.hparams.teacher_model
        if self.hparams.seq_pad:
            self.pad_x = nn.ZeroPad2d((0,0,1,1))
            self.pad_mask = nn.ConstantPad2d((0,0,2,0),1.0)
            self.pad_list = nn.ZeroPad2d((1,1,0,0))

    def forward(self, x,  padding_mask, gene_indicators, sv_indicators):
        """
        Inputs:
        x- Input features of shape [Batch, SeqLen, input_dim], within SV
        MHA_mask - Mask to apply on the attention outputs in MHA (optional)
        gene_indicators - gene indicators for x1 of shape [Batch, SeqLen]
        disease_gene_scores - optional of shape[Batch, SeqLen]
        """
        if self.hparams.seq_pad:
            x = self.pad_x(x)
            padding_mask = self.pad_mask(padding_mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18

        instance_embedding = self.feature_extractor(x,  padding_mask, gene_indicators, sv_indicators)
        if self.hparams.indicator_type=='sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators
        #if attention dim =-3, doesn't matter the choise of indicator, output instance y_logit
        y_logit, instance_weights = self.classifier(gene_embedding=instance_embedding,sv_indicators=indicator)#pull by sv_indicator
        y_pred = torch.sigmoid(y_logit) # no temperature for inference
        return y_pred, y_logit, instance_weights, instance_embedding

    @torch.no_grad()
    def get_attention_maps(self, x, gene_indicators, sv_indicators):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        """
        if self.hparams.seq_pad:
            x = self.pad_x(x)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)  # 12/18
        attention_maps = self.feature_extractor.get_attention_maps(x, gene_indicators, sv_indicators)
        return attention_maps

    @torch.no_grad()
    def get_element_score(self, batch):
        #use for local model

        x, mask, gene_indicators, sv_indicators, y= batch

        if self.hparams.seq_pad:
            x = self.pad_x(x)
            mask = self.pad_mask(mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18

        if self.hparams.indicator_type=='sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators

        instance_embedding = self.feature_extractor(x, mask, gene_indicators, sv_indicators)
        _, seq_len, _ = instance_embedding.size()

        #elements
        element_predictions=[]
        for index in range(seq_len):
            instance_embedding_ = instance_embedding[:, index:(index + 1), :]
            y_logit_, _ = self.classifier(gene_embedding=instance_embedding_,sv_indicators=indicator[:, index:(index + 1)])

            element_prediction = torch.sigmoid(y_logit_).item()
            element_predictions.append(element_prediction)
        #overall
        y_logit, _ = self.classifier(gene_embedding=instance_embedding,sv_indicators=indicator)
        prediction = torch.sigmoid(y_logit).item()
        truth = y.int().item()

        return prediction, truth, element_predictions


    def configure_optimizers(self):
        if self.hparams.optimizer=='adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        # Apply lr scheduler per step
        lr_scheduler = None
        if self.hparams.scheduler=='multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.hparams.max_iters / 3, self.hparams.max_iters / 3 * 2], gamma=0.1)
        elif self.hparams.scheduler=='cos':
            lr_scheduler = of.CosineWarmupScheduler(optimizer,warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]


    def training_step(self, batch, batch_idx):
        x, mask, gene_indicators, sv_indicators, y = batch

        if self.hparams.indicator_type=='sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators


        if self.hparams.seq_pad:
            x = self.pad_x(x)
            mask = self.pad_mask(mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18
            indicator = self.pad_list(indicator)

        instance_embedding = self.feature_extractor(x, mask, gene_indicators,sv_indicators)
        y_logit, _  = self.classifier(gene_embedding=instance_embedding, sv_indicators=indicator)

        if self.teacher_model is not None and self.current_epoch>=1:
            teacher_model=os.path.join(self.teacher_model, os.listdir(self.teacher_model)[0])
            teacher_mod = self.load_from_checkpoint(teacher_model).to(self.device)
            teacher_mod.eval()
            _, teacher_logit, _, _= teacher_mod(x,  mask, gene_indicators = gene_indicators, sv_indicators=indicator)
            teacher_pred = torch.sigmoid(teacher_logit/self.hparams.temperature).to(self.device)
            loss = self.hparams.beta * of.soft_cce_loss(y_logit, teacher_pred.detach(),reduction='none')+ \
                   (1- self.hparams.beta)*self.criterion(y_logit, y)
            loss = torch.mean(loss)
        else:
            loss = torch.mean(self.criterion(y_logit.unsqueeze(1), y.unsqueeze(1)))

        acc = self.accuracy(torch.sigmoid(y_logit),y.int())

        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist = True)
        self.log('train_loss',loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, mask, gene_indicators, sv_indicators, y = batch

        if self.hparams.indicator_type == 'sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators
        if self.hparams.seq_pad:
            x = self.pad_x(x)
            mask = self.pad_mask(mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18
            indicator = self.pad_list(indicator)

        instance_embedding = self.feature_extractor(x, mask, gene_indicators, sv_indicators)
        y_logit, _ = self.classifier(gene_embedding=instance_embedding, sv_indicators=indicator)


        loss = torch.mean(self.criterion(y_logit, y))
        acc = self.accuracy(torch.sigmoid(y_logit), y.int())

        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist = True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist = True)


    def test_step(self, batch, batch_idx):
        x, mask, gene_indicators, sv_indicators, y = batch

        if self.hparams.indicator_type == 'sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators
        if self.hparams.seq_pad:
            x = self.pad_x(x)
            mask = self.pad_mask(mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18
            indicator = self.pad_list(indicator)

        instance_embedding = self.feature_extractor(x, mask, gene_indicators, sv_indicators)
        y_logit, _ = self.classifier(gene_embedding=instance_embedding, sv_indicators=indicator)

        loss = torch.mean(self.criterion(y_logit, y))
        acc = self.accuracy(torch.sigmoid(y_logit), y.int())

        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True)


    def predict_step(self, batch, batch_idx):
        x, mask, gene_indicators, sv_indicators, y = batch

        if self.hparams.indicator_type == 'sv':
            indicator = sv_indicators
        else:
            indicator = gene_indicators
        if self.hparams.seq_pad:
            x = self.pad_x(x)
            mask = self.pad_mask(mask)
            gene_indicators = self.pad_list(gene_indicators)
            sv_indicators = self.pad_list(sv_indicators)#12/18
            indicator = self.pad_list(indicator)

        instance_embedding = self.feature_extractor(x, mask, gene_indicators,sv_indicators)
        y_logit, _ = self.classifier(gene_embedding=instance_embedding, sv_indicators=indicator)

        prediction = torch.sigmoid(y_logit).item()
        truth = y.int().item()
        return prediction, truth


