#end to end training of net for meso
from scipy.sparse.construct import rand
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import AveragePrecision, ROC_AUC, RocCurve, PrecisionRecallCurve
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import pandas as pd
from mk_blocks import stratified_split
from df_to_hdf5 import df_to_hdf5, to_binary_class
from hdf5_torchdset import hdf5Dataset
import pickle
import av_agg_pred_metric as agg
from ignite.contrib.handlers.tensorboard_logger import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
from mpl_toolkits.axes_grid1 import ImageGrid
from ignite.handlers import EarlyStopping
#import os
import torchvision.transforms.functional as TF
import random
from MIL_sampler import WeightedMILSampler, WeightedRandomStratifiedSampler
#from tcga_utilities import load_model, hed_grey_transform

#Train a variety of models on TCGA or MESO TMAs using pure patchwise or MIL methods

#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
pre_stage='none' #sarc, normal, none
MIL_type='sample' #sample, batch or none
batch_size=64
nsample=30   #for batch MIL type
agg_topN='mean'
epochs=400
sample_factor=1   #0.6  for sample MIL type
num_workers=6    #def 6
lr=0.0005
wd=0.0001
same_per_slide=False
extra_info='sample, adam'

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class hed_random:
    def __init__(self, probs):
        self.probs = probs

    def __call__(self, x):
        zerod=0
        for i in range(2,-1,-1):
            r=random.uniform(0.0,1.0)
            if r<self.probs[i]:
                x[i,:,:]=0
                zerod=zerod+1
            if zerod==2:
                break

        return x

class best_tracker():
    def __init__(self,trainer) -> None:
        self.best_score=0
        self.best_epoch=0
        self.trainer=trainer
    def update(self,this_score):
        if this_score>self.best_score and self.trainer.state.epoch>3:
            self.best_epoch=self.trainer.state.epoch
            self.best_score=this_score
    def new_trainer(self,trainer):
        self.trainer=trainer

class alpha_store():
    def __init__(self, alpha, step_size) -> None:
        self.alpha=alpha
        self.step_size=step_size
    
    def update_alpha(self):
        if self.alpha<0.9:
            self.alpha+=self.step_size
        else:
            self.alpha=0.9

class im_trans():
    def __call__(self, im):
        im=np.float32(im)/255
        return im

class lab_trans():
    def __call__(self, label):
        label=np.int64(label)
        return label

class input_means():
    def __init__(self) -> None:
        self.means=[]
    
    def __call__(self, mean):
        self.means.append(mean)

def get_data_loaders(split_df=None, base_path=None, fracs=[0.6, 0.2, 0.2]):
    
    sampler=None
    norm=transforms.Normalize(mean=[0.851, 0.706 , 0.803],std=[0.0776,0.110,0.0813])   
    
    trans_tr=transforms.Compose([im_trans(),
        torch.from_numpy,
        transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        RotationTransform(angles=[-90, 0, 90, 180]),
        norm])
    trans_tr_val=transforms.Compose([im_trans(),
        torch.from_numpy,
      norm])
    trans_val=transforms.Compose([im_trans(),
        torch.from_numpy,
      norm])                        
    tar_trans=lab_trans()
    

    if split_df==None:  
        with open(Path('E:\TCGA_Data\MESOv_hdf5\TMA_labels.pkl'), 'rb') as loadfile:
            unpickler = pickle.Unpickler(loadfile)
            df=unpickler.load()
            loadfile.close()
            dfs=stratified_split(df.iloc[52:], fracs=[0.75,0.25,0], shuffle=True)
            dfs['Test']=df.iloc[0:52]       
    else:
        dfs=split_df

    sarc_only_df=None
    if pre_stage=='sarc':
        sarc_only_df=dfs['Train'][dfs['Train'].labels!='Biphasic']


    tr_labs=[]
    for lab in dfs['Train']['labels']:
        tr_labs.append(to_binary_class(lab))
    class_weights=compute_class_weight('balanced', classes=[0,1], y=tr_labs)
    class_ratio=(len(tr_labs)-sum(tr_labs))/sum(tr_labs)

    with open(base_path.joinpath('dfs.pkl'), 'wb') as output:
        pickler = pickle.Pickler(output, -1)
        pickler.dump(dfs)
        output.close()
    info_df=pd.DataFrame({'Parameter': ['pre_stage','MIL_type','batch_size','epochs','lr','nsample',
        'agg_topN','wd','same_per_slide','sample_factor','num_workers','extra_info'],
        'Value': [pre_stage,MIL_type,batch_size,epochs,lr,nsample,agg_topN,wd,same_per_slide,sample_factor,num_workers,extra_info] })
    info_df.to_csv(base_path.joinpath('info.csv'))

    data_loaders={}
    for key in dfs.keys():
        hdf_path=base_path.joinpath(f'temp_{key}.hdf5')
        if pre_stage=='sarc' and key=='Train':
            df_to_hdf5(sarc_only_df,hdf_path,1200,use_all=True)
        else:
            df_to_hdf5(dfs[key],hdf_path,1200,use_all=True)
        
        if key=='Train':
            ds=hdf5Dataset(hdf_path,trans_tr,tar_trans)
            h_weights=ds.get_weights()
            NS=int(np.maximum(sample_factor,sample_factor*len(ds)))
            if pre_stage=='sarc':
                labs=ds.get_labels()
                argh=sum(h_weights[labs==0])/(sum(h_weights[labs==1])*class_ratio)
                h_weights[labs==1]=argh*h_weights[labs==1]
                sampler=WeightedRandomStratifiedSampler(h_weights, int(0.2*NS), sizes=ds.sizes, same_per_slide=same_per_slide)
            elif MIL_type!='batch' or pre_stage=='normal':
                sampler=WeightedRandomStratifiedSampler(h_weights, NS, sizes=ds.sizes, same_per_slide=same_per_slide)
            else:
                sampler=WeightedMILSampler(h_weights, 70, sizes=ds.sizes, btch_size=batch_size)
            data_loaders[key]=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, pin_memory=True)
            hdf_path=base_path.joinpath('temp_Tr_eval.hdf5')
            df_to_hdf5(dfs[key],hdf_path,1200,use_all=True, epoch=1)
            ds=hdf5Dataset(hdf_path,trans_tr_val,tar_trans)
            data_loaders['Tr_eval']=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            ds=hdf5Dataset(hdf_path,trans_val,tar_trans)
            data_loaders[key]=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        

    return data_loaders, dfs, class_weights, class_ratio, sarc_only_df

def activated_output_transform(output):
    y_pred, y = output
    y_pred=F.softmax(y_pred,1)[:,1]
    return y_pred, y

def agg_transform_fn(output, sizes):
    csum_sizes=np.cumsum(sizes)
    y_pred, y = output
    y_pred=F.softmax(y_pred,1)
    y_pred_out, y_out=torch.zeros([0,2], dtype=y_pred.dtype, device='cuda'), y[0]
    for i, size in enumerate(sizes):
        if i==0:
            torch.cat((y_pred_out,torch.unsqueeze(torch.mean(y_pred[0:csum_sizes[i],:],0),0)),0)
        else:
            y_pred_out.append(torch.mean(y_pred[csum_sizes[i-1]:csum_sizes[i],:],0))
            y_out.append(y[sizes[i]-1])

    y_pred_out=y_pred_out[:,1]
    return y_pred_out, y_out

def log_roc_image(preds, labels, auc, logger, tag, global_step):
    fpr, tpr, thresholds=roc_curve(labels,preds, pos_label=1)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot()
    logger.writer.add_figure(tag,display.figure_,global_step)

def log_pr_image(preds, labels, ap, logger, tag, global_step):
    prec, recall, thresholds=precision_recall_curve(labels, preds, pos_label=1)
    display=PrecisionRecallDisplay(precision=prec, recall=recall, average_precision=ap)
    display.plot()
    logger.writer.add_figure(tag,display.figure_,global_step)

def pind_2_sind(ind,dl):
    cum_sizes=np.cumsum(dl.dataset.sizes)
    return sum(ind.item()>cum_sizes)

def log_top_images(inds,dl,split, dfs, trainer, tb_logger):
    #5 most sarcomatoid images then 5 least
    imgs, labs=[],[]
    dl.dataset.init_vds()
    for i in inds:
        imgs.append(np.transpose(dl.dataset.ds[i,:,:,:],(1,2,0)))
        s_ind=pind_2_sind(i,dl)
        labs.append(dfs[split]['labels'].iloc[s_ind])

    fig = plt.figure(figsize=(16., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 8),  # creates 2x2 grid of axes
                 axes_pad=(0.01,0.25),  # pad between axes in inch.
                 )

    for ax, im, lab in zip(grid, imgs, labs):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(lab, pad=0.01)

    tb_logger.writer.add_figure(f'{split}/top_instances',fig,trainer.state.epoch)
    plt.close('all')


def train_net(split_df=None, base_path=Path('D:\TCGA_Data\Temp'), back_model=None):
    
    from batch_MIL import BatchMIL_CE_Loss
    from tiatoolbox.models.classification import CNNPatchPredictor
    matplotlib.use("Agg")

    model=torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    top = nn.Sequential(
                nn.Linear(num_ftrs, 2)
                )
    model.fc=top
    for param in model.parameters():
        param.requires_grad = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    model.to(device)

    data_loaders, dfs, class_weights, class_ratio, sarc_only_df = get_data_loaders(split_df, base_path)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, weight_decay=wd, momentum=0.8, nesterov=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=wd)
    cycle_scheduler = OneCycleLR(optimizer,max_lr=lr, steps_per_epoch=len(data_loaders['Tr_eval']), epochs=epochs, pct_start=0.3, div_factor=20, final_div_factor=100)
    scheduler = LRScheduler(cycle_scheduler)

    criterion2 = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = BatchMIL_CE_Loss(weight=torch.Tensor(class_weights).to(device))
    alpha=alpha_store(0.0, 1/(10*len(data_loaders['Train'])))
    phase=0

    def train_step(engine, batch):
        model.train()
        X,y=batch[0],batch[1]
        y_pred = model(X.to(device, non_blocking=True))
        
        if phase==0 or (MIL_type!='batch'):
            loss = criterion2(y_pred, y.to(device))
        else:
            loss = criterion(y_pred, y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    trainer = Engine(train_step)

    tr_metrics = {
        "accuracy": Accuracy(),
        "cel": Loss(criterion2),
        "ap": AveragePrecision(activated_output_transform),
        "auc": ROC_AUC(activated_output_transform),
        "roc_c": RocCurve(activated_output_transform),
        "prc": PrecisionRecallCurve(activated_output_transform),
        "agg_auc": agg.av_aggregation_metric(data_loaders['Train'].dataset.sizes, agg_topN, agg.auc_roc_compute_fn),
        "agg_ap": agg.av_aggregation_metric(data_loaders['Train'].dataset.sizes, agg_topN, agg.ap_prc_compute_fn)
    }
    val_metrics = {
        "accuracy": Accuracy(),
        "cel": Loss(criterion2),
        "ap": AveragePrecision(activated_output_transform),
        "auc": ROC_AUC(activated_output_transform),
        "roc_c": RocCurve(activated_output_transform),
        "prc": PrecisionRecallCurve(activated_output_transform),
        "agg_auc": agg.av_aggregation_metric(data_loaders['Val'].dataset.sizes, agg_topN, agg.auc_roc_compute_fn),
        "agg_ap": agg.av_aggregation_metric(data_loaders['Val'].dataset.sizes, agg_topN, agg.ap_prc_compute_fn)
    }
    test_metrics = {
        "accuracy": Accuracy(),
        "cel": Loss(criterion2),
        "ap": AveragePrecision(activated_output_transform),
        "auc": ROC_AUC(activated_output_transform),
        "roc_c": RocCurve(activated_output_transform),
        "prc": PrecisionRecallCurve(activated_output_transform),
        "agg_auc": agg.av_aggregation_metric(data_loaders['Test'].dataset.sizes, agg_topN, agg.auc_roc_compute_fn),
        "agg_ap": agg.av_aggregation_metric(data_loaders['Test'].dataset.sizes, agg_topN, agg.ap_prc_compute_fn)
    }
    evaluator_tr = create_supervised_evaluator(model, metrics=tr_metrics, device=device)#, non_blocking=True)
    evaluator_val = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    evaluator_test = create_supervised_evaluator(model, metrics=test_metrics, device=device)
    tracker=best_tracker(trainer)

    tb_logger = TensorboardLogger(log_dir=base_path.joinpath('tb_logs'))
    to_log=['accuracy','cel','auc','ap','agg_auc','agg_ap']

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=10),
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )

    tb_logger.attach_output_handler(
        evaluator_tr,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=to_log,
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        evaluator_val,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=to_log,
        global_step_transform=global_step_from_engine(trainer),
    )

    def ap_plus_auc(engine):
        ap=engine.state.metrics['agg_ap']
        auc=engine.state.metrics['agg_auc']
        if trainer.state.epoch>5:
            return 0.25*ap+auc
        else:
            return (0.25*ap+auc)*0.5
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def update_alpha(trainer):
        alpha.update_alpha()

    @trainer.on(Events.ITERATION_COMPLETED(every=25))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f} alpha: {alpha.alpha:.3f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator_tr.run(data_loaders['Tr_eval'])
        metrics = evaluator_tr.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['cel']:.2f} ap: {metrics['ap']:.3f} auc: {metrics['auc']:.3f} agg_ap: {metrics['agg_ap']:.3f} agg_auc: {metrics['agg_auc']:.3f}")
        preds, labels=tr_metrics['agg_ap'].get_data()
        preds=F.softmax(preds,1)[:,1]
        log_pr_image(preds, labels, metrics['ap'], tb_logger, 'tile/pr_curve_tr', trainer.state.epoch)
        log_roc_image(preds, labels, metrics['auc'], tb_logger, 'tile/roc_tr', trainer.state.epoch)
        inds=np.argsort(preds)
        inds=inds[[-1,-2,-3,-4,-5,-6,-7,-8,0,1,2,3,4,5,6,7]]
        log_top_images(inds,data_loaders['Tr_eval'],'Train',  dfs, trainer, tb_logger)
        
        preds, labels=tr_metrics['agg_ap'].get_data(agg=True)
        log_pr_image(preds, labels, metrics['agg_ap'], tb_logger, 'slide/pr_curve_tr', trainer.state.epoch)
        log_roc_image(preds, labels, metrics['agg_auc'], tb_logger, 'slide/roc_tr', trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator_val.run(data_loaders['Val'])
        metrics = evaluator_val.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['cel']:.2f} ap: {metrics['ap']:.3f} auc: {metrics['auc']:.3f} agg_ap: {metrics['agg_ap']:.3f} agg_auc: {metrics['agg_auc']:.3f}")
        preds, labels=val_metrics['agg_ap'].get_data()
        preds=F.softmax(preds,1)[:,1]
        log_pr_image(preds, labels, metrics['ap'], tb_logger, 'tile/pr_curve_val', trainer.state.epoch)
        log_roc_image(preds, labels, metrics['auc'], tb_logger, 'tile/roc_val', trainer.state.epoch)
        inds=np.argsort(preds)
        inds=inds[[-1,-2,-3,-4,-5,-6,-7,-8,0,1,2,3,4,5,6,7]]
        log_top_images(inds,data_loaders['Val'],'Val',  dfs, trainer, tb_logger)
        
        preds, labels=val_metrics['agg_ap'].get_data(agg=True)
        log_pr_image(preds, labels, metrics['agg_ap'], tb_logger, 'slide/pr_curve_val', trainer.state.epoch)
        log_roc_image(preds, labels, metrics['agg_auc'], tb_logger, 'slide/roc_val', trainer.state.epoch)

        tracker.update(ap_plus_auc(evaluator_val))

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_datasets(trainer):
        print('updating datasets')

        for key in data_loaders.keys():
            if key=='Test': continue
            hdf_path=base_path.joinpath(f'temp_{key}.hdf5')
            data_loaders[key].dataset.close_hdf()
            if key=='Tr_eval':
                df_to_hdf5(dfs['Train'],hdf_path,1200, use_all=True, epoch=trainer.state.epoch+1)
            elif key=='Train' and pre_stage=='sarc' and phase==0:
                df_to_hdf5(sarc_only_df,hdf_path,1200, use_all=True, epoch=trainer.state.epoch)
            else:
                df_to_hdf5(dfs[key],hdf_path,1200, use_all=True, epoch=trainer.state.epoch)
            data_loaders[key].dataset.refresh_vds()
            if key=='Train':
                if MIL_type=='sample' and phase==1:
                    preds,_=tr_metrics['agg_ap'].get_data()    #val metrics??
                    preds=F.softmax(preds,1)[:,1]
                    print('getting weights..')
                    h_weights=data_loaders[key].dataset.get_weights(preds)
                else:
                    h_weights=data_loaders[key].dataset.get_weights()
                if pre_stage=='sarc' and phase==0:
                    labs=data_loaders[key].dataset.get_labels()
                    argh=sum(h_weights[labs==0])/(sum(h_weights[labs==1])*class_ratio)
                    h_weights[labs==1]=argh*h_weights[labs==1]
                data_loaders[key].sampler.weights=torch.as_tensor(h_weights, dtype=torch.double)
            elif key=='Val' and MIL_type=='sample' and phase==1:
                v_preds,_=val_metrics['agg_ap'].get_data()    #val metrics??
                v_preds=F.softmax(preds,1)[:,1]
                data_loaders[key].dataset.get_weights(v_preds)


    score_function = ap_plus_auc

    to_save = {'model': model, 'eval_tr': evaluator_tr, 'eval_val': evaluator_val}
    handler2 = Checkpoint(
        to_save, DiskSaver(base_path.joinpath('models'), create_dir=True),
        n_saved=2, filename_prefix='best',
        score_function=score_function, score_name="val_ap_auc",
        global_step_transform=global_step_from_engine(trainer)
    )

    evaluator_val.add_event_handler(Events.COMPLETED, handler2)

    gst = lambda *_: trainer.state.epoch
    handler1=Checkpoint(to_save, DiskSaver(base_path.joinpath('models'), create_dir=True), n_saved=5, global_step_transform=gst)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler1)

    #do early stopping
    handler = EarlyStopping(patience=40, score_function=score_function, trainer=trainer)
    evaluator_val.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.TERMINATE | Events.COMPLETED)
    def log_testing_results(trainer):
        if trainer.state.epoch>5:
            #save weight history for vis
            np.savez(base_path.joinpath('w_hist.npz'), wh=np.vstack(data_loaders['Train'].dataset.weight_hist), 
                sizes=data_loaders['Train'].dataset.get_sizes(), rh=np.vstack(data_loaders['Train'].dataset.raw_hist),
                labs=data_loaders['Train'].dataset.get_labels())
            np.savez(base_path.joinpath('w_hist_val.npz'), wh=np.vstack(data_loaders['Val'].dataset.weight_hist), 
                sizes=data_loaders['Val'].dataset.get_sizes(), rh=np.vstack(data_loaders['Val'].dataset.raw_hist),
                labs=data_loaders['Val'].dataset.get_labels())

            #recover best model according to val set
            print(f'loading best epoch: {tracker.best_epoch}')
            checkpoint_fp=base_path.joinpath('models',f'best_checkpoint_{tracker.best_epoch}_val_ap_auc={tracker.best_score:.4f}.pt')
            checkpoint = torch.load(checkpoint_fp)
            to_load = {"model": model}
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

            evaluator_test.run(data_loaders['Test'])
            metrics = evaluator_test.state.metrics
            print(f"Test Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['cel']:.2f} ap: {metrics['ap']:.3f} auc: {metrics['auc']:.3f} agg_ap: {metrics['agg_ap']:.3f} agg_auc: {metrics['agg_auc']:.3f}")
            preds, labels=test_metrics['agg_ap'].get_data()
            preds=F.softmax(preds,1)[:,1]
            log_pr_image(preds, labels, metrics['ap'], tb_logger, 'tile/pr_curve_test', trainer.state.epoch)
            log_roc_image(preds, labels, metrics['auc'], tb_logger, 'tile/roc_test', trainer.state.epoch)
            inds=np.argsort(preds)
            inds=inds[[-1,-2,-3,-4,-5,-6,-7,-8,0,1,2,3,4,5,6,7]]
            log_top_images(inds,data_loaders['Test'],'Test',  dfs, trainer, tb_logger)
            
            preds, labels=test_metrics['agg_ap'].get_data(agg=True)
            log_pr_image(preds, labels, metrics['agg_ap'], tb_logger, 'slide/pr_curve_test', trainer.state.epoch)
            log_roc_image(preds, labels, metrics['agg_auc'], tb_logger, 'slide/roc_test', trainer.state.epoch)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

    if pre_stage=='none':
        phase=1
        trainer.run(data_loaders['Train'], max_epochs=epochs)
    else:
        trainer.run(data_loaders['Train'], max_epochs=5)
        if pre_stage=='sarc':
            hdf_path=base_path.joinpath('temp_Train.hdf5')
            data_loaders['Train'].dataset.close_hdf()
            df_to_hdf5(dfs['Train'],hdf_path,1200, use_all=True, epoch=trainer.state.epoch)
            ds=hdf5Dataset(hdf_path,data_loaders['Train'].dataset.transform,data_loaders['Train'].dataset.target_transform)
            h_weights=ds.get_weights()
            if MIL_type!='batch':
                sampler=WeightedRandomStratifiedSampler(h_weights, int(np.max(sample_factor,sample_factor*len(ds))), sizes=ds.sizes, same_per_slide=same_per_slide)
            else:
                sampler=WeightedMILSampler(h_weights, 70, sizes=ds.sizes, btch_size=batch_size)
            data_loaders['Train']=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, pin_memory=True)
        elif MIL_type=='batch':
            ds=data_loaders['Train'].dataset
            h_weights=ds.get_weights()
            sampler=WeightedMILSampler(h_weights, 70, sizes=ds.get_sizes(), btch_size=batch_size)
            data_loaders['Train']=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, pin_memory=True)
        
        phase=1
        print('starting MIL phase')
        trainer.state.max_epochs = None
        trainer.set_data(data_loaders['Train'])
        trainer.run(data_loaders['Train'], max_epochs=epochs)
    
    tb_logger.close()

if __name__=='__main__':
    train_net()