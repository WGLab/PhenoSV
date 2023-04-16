import os
import sys
import argparse
import pandas as pd
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import model.model as mm
import model.data_loader as dat
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint




parser = argparse.ArgumentParser(description='SV TRANSFORMER MODEL')
#dataset
parser.add_argument('--result_dir', type=str, default='/athena/marchionnilab/scratch/zhx2006/results/')
parser.add_argument('--df_path_train', type=str, default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/train_clinvar.csv')
parser.add_argument('--df_path_validation', type=str, default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/validation_clinvar.csv')
parser.add_argument('--df_path_test', type=str, default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/test_clinvar.csv')
parser.add_argument('--feature_path', type=str, default='/athena/marchionnilab/scratch/zhx2006/sv_feature/local/')
parser.add_argument('--df_path_train_domain', type=str)#set when train sv-gene dann
parser.add_argument('--df_path_validation_domain', type=str)
parser.add_argument('--number_layers_conv',type=int, default=3)
#train method
parser.add_argument('--seed', type=int, default=46)
parser.add_argument('--sampler_train', type=str)#col name for sample weights; change to CLASS_SAMPLE_WEIGHT for only class balance, change to None to cancle sampler
parser.add_argument('--sampler_vali', type=str)
parser.add_argument('--sampler_domain', type=str)
parser.add_argument('--pos_weight',type=float)#weight for positive cases to add in loss
parser.add_argument('--pretrained_model',type=str,default=None)#set noncoding bin for sv_gene model
parser.add_argument('--freeze_pretrain', action='store_true')
parser.add_argument('--unfreeze_pretrain',dest='freeze_pretrain',action='store_false')
parser.add_argument('--finetune_last', action='store_true')
parser.add_argument('--free_classifier', dest='finetune_last', action='store_false')
parser.add_argument('--train_samples', type=int, default=10000)
parser.add_argument('--vali_samples', type=int,default=None)
parser.add_argument('--all_features', action='store_true')
parser.add_argument('--all_features_off',dest='all_features',action='store_false')#turn off all features to use feature subset, set feature_group
parser.add_argument('--feature_group',type=str)
parser.add_argument('--feature_grouping_path', type=str)#leave it empty if not grouping
parser.add_argument('--feature_index',type=str)#use to explicitly choose subset of features: '12,13,14,15,16,61,-1'
parser.add_argument('--indicator_type', type=str,default='gene')
parser.add_argument('--seq_pad', action='store_true')
parser.add_argument('--seq_pad_off', action='store_false',dest='seq_pad')
parser.add_argument('--clip_pad', action='store_true')#clip relative positional embedding for padding
parser.add_argument('--clip_pad_off', action='store_false',dest='clip_pad')
#model structure
parser.add_argument('--input_dim', type=int, default=271)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_middle_dim', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--number_layers_encoder', type=int, default=2)
parser.add_argument('--attention_dim', type=int, default=16)#set to -1 to use sentence embedding, 0: maxpooling, positive: MIL
parser.add_argument('--classifier_hidden', type=int, default=128)
parser.add_argument('--max_relative_position',type=int, default=0)#set to a positive value to use relative positional encoding
#learning parameters
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str,default='adam')
parser.add_argument('--scheduler', type=str, default='multistep')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--warmup', type=int, default=50)#for cosine lr scheduler
parser.add_argument('--max_iters', type=int, default=100)#epoch
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--temperature', type=float, default=1.0)#teacher model temperature
parser.add_argument('--distill', action='store_true')
parser.add_argument('--distill_off',dest='distill',action='store_false')
parser.add_argument('--beta', type=float, default=0.5)#distillation
parser.add_argument('--loader_workers', type=int, default=4)
parser.add_argument('--accumulate_batches', type=int, default=0)
parser.add_argument('--disable_progress_bar', action='store_true')
parser.add_argument('--enable_progress_bar', dest='disable_progress_bar',action='store_false')

def main():
    global args
    args = parser.parse_args()
    default_root_dir = os.path.join(args.result_dir, str(datetime.datetime.now()))
    os.makedirs(default_root_dir)
    print('results folder is: ' + str(default_root_dir))
    if args.distill:
        teacher_path = os.path.join(default_root_dir, "lightning_logs/0/checkpoints")
    else:
        teacher_path = None
    pl.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('device number: ')
        print(torch.cuda.device_count())
        print('device name: ')
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    #--------------init the model
    setting = {'knowledge_distillation':args.distill,
               'train': args.df_path_train, 'sampler_train': args.sampler_train, 'train_sample_num': args.train_samples,
               'train_domain': args.df_path_train_domain,'sampler_domain':args.sampler_domain,
               'validation': args.df_path_validation,'validation_domain':args.df_path_validation_domain,
               'sampler_vali':args.sampler_vali,'validation_sample_num': args.vali_samples,
               'test': args.df_path_test,'feature_path':args.feature_path,
               'batch_size':args.batch_size, 'temperature':args.temperature,
               'pretrained_model': args.pretrained_model,'freeze_pretrain':args.freeze_pretrain,
               'accumulate_batches':args.accumulate_batches, 'all_features': args.all_features,
               'seed':args.seed, 'feature_group':args.feature_group, 'feature_grouping_path':args.feature_grouping_path}
    setting_df = pd.DataFrame([setting]).transpose()
    setting_df.to_csv(os.path.join(default_root_dir, 'setting.csv'))
    print('the setting is: '+ str(setting))



    if args.all_features==False:
        assert [args.feature_grouping_path,args.feature_index].count(None)<2,"feature_grouping_path or feature_index cannot be both None, add path four feature subset"
        if args.feature_grouping_path is not None:
            df = pd.read_csv(args.feature_grouping_path)
            if ',' in args.feature_group:
                feature_group = [int(i) for i in args.feature_group.split(',')]
            else:
                feature_group=[int(args.feature_group)]
            feature_group_index = df[df['group'].isin(feature_group)]['index'].tolist()
            feature_group_index.append(-1)
        else:
            if ',' in args.feature_index:
                feature_group_index = [int(i) for i in args.feature_index.split(',')]
            else:
                feature_group_index = args.feature_index
    else:
        feature_group_index=None
    print('feature_group_index: '+str(feature_group_index))

    #coding model: 'local'
    hparams = {'input_dim': args.input_dim, 'input_middle_dim': args.input_middle_dim,
                   'model_dim': args.model_dim, 'num_heads': args.num_heads,
               'number_layers_encoder': args.number_layers_encoder,
               'attention_dim': args.attention_dim, 'classifier_hidden': args.classifier_hidden,
               'lr': args.lr, 'warmup': args.warmup, 'max_iters': args.max_iters, 'dropout': args.dropout,
               'weight_decay': args.weight_decay, 'temperature': args.temperature, 'scheduler': args.scheduler,
               'beta':args.beta,'teacher_model': teacher_path, 'optimizer': args.optimizer,
                     'indicator_type':args.indicator_type,'seq_pad':args.seq_pad,
                     'max_relative_position':args.max_relative_position,'pos_weight':args.pos_weight,
                     'clip_pad':args.clip_pad}


    # --------------init model
    hparams_df = pd.DataFrame([hparams]).transpose()
    hparams_df.to_csv(os.path.join(default_root_dir, 'hparams.csv'))
    print('parameters are: ' + str(hparams))

    print('initiate model')

    if args.pretrained_model is not None:
        model = mm.LocalSVModule.load_from_checkpoint(args.pretrained_model,**hparams)
        if args.freeze_pretrain:
            for child in model.feature_extractor.children():
                for child1 in child.children():
                    for param in child1.parameters():
                        param.requires_grad = False
        if args.finetune_last:
            for child in model.classifier.children():
                for i,child1 in enumerate(child.children()):
                    if i<2:
                        for param in child1.parameters():
                            param.requires_grad = False
    else:
        model = mm.LocalSVModule(**hparams)
        print('train local model')

    # --------------init dataset
    if torch.cuda.device_count()>1:
        distributed = True
    else:
        distributed = False


    train_df = pd.read_csv(args.df_path_train)
    train_df['PATH'] = [os.path.join(args.feature_path ,i+'.npz') for i in train_df['ID'].tolist()]
    vali_df = pd.read_csv(args.df_path_validation)
    vali_df['PATH'] = [os.path.join(args.feature_path, i + '.npz') for i in vali_df['ID'].tolist()]
    test_df = pd.read_csv(args.df_path_test)
    test_df['PATH'] = [os.path.join(args.feature_path, i + '.npz') for i in test_df['ID'].tolist()]

    DATA = dat.DataModule(train_df=train_df, vali_df=vali_df, test_df=test_df, batch_size=args.batch_size,
                                  train_sampler_col=args.sampler_train, vali_sampler_col=args.sampler_vali,
                                  num_samples=args.train_samples, vali_samples=args.vali_samples,
                                  loader_workers=args.loader_workers, distributed=distributed,
                                  perm=None, feature_subset=feature_group_index)

    # --------------set up trainer
    print('set up trainer')
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(default_root_dir, 'logger_tb'))
    csv_logger = pl.loggers.CSVLogger(save_dir=os.path.join(default_root_dir, 'logger_csv'))
    save_best = ModelCheckpoint(dirpath=default_root_dir, filename="model-{epoch:02d}-{val_loss:.2f}",
                                save_top_k=1, save_weights_only=True, mode="min", monitor="val_loss")
    save_last_epoch = ModelCheckpoint(save_weights_only=True, filename="model-teacher")
    if args.disable_progress_bar:
        progressbar=False
    else:
        progressbar=True

    if args.sampler_train is None:
        replace_sampler_ddp=True
    else:
        replace_sampler_ddp=False

    trainer = pl.Trainer(default_root_dir=default_root_dir,
                         max_epochs=args.max_iters,
                         auto_select_gpus=True,
                         accelerator='gpu',
                         devices=int(torch.cuda.device_count()),
                         enable_progress_bar=progressbar,
                         callbacks=[save_best,save_last_epoch],
                         logger=[tensorboard_logger, csv_logger],
                         replace_sampler_ddp=replace_sampler_ddp,
                         log_every_n_steps=int(args.train_samples/args.batch_size),
                         accumulate_grad_batches =int(args.accumulate_batches))



    #---------------fit model
    print('start training model')
    trainer.fit(model, DATA)
    #----------------test model
    print('evaluating on test dataset')
    bestmodel_path = [os.path.join(default_root_dir, i) for i in os.listdir(default_root_dir) if 'ckpt' in i][0]
    print('the best model path is: '+ str(bestmodel_path))

    model = mm.LocalSVModule.load_from_checkpoint(bestmodel_path)
    test_result = trainer.test(model, DATA, verbose=False)
    result = {"test": test_result[0]["test_acc"]}
    print(result)


if __name__ == '__main__':
    main()
