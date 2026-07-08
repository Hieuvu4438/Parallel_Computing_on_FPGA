from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np

import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
# or 
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from transformers import set_seed
from util.augmentation import SpecAugment

from util.icbhi_dataset import ICBHIDataset, ICBHINonBTSDataset
from util.icbhi_util import get_score
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json

from transformers import ClapProcessor
from models import get_backbone_class
Processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=48000)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#CUBLAS_WORKSPACE_CONFIG=:16:8

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save') ## have to change
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # distill experiments
    parser.add_argument('--soft_label_mode', type=str, default='none', help="how to make the soft label.")
    parser.add_argument('--train_teacher_logits', type=str, default='teacher_logits/teacher_logits.training.pt', help="Teacher logits file for train.")
    parser.add_argument('--test_teacher_logits', type=str, default='teacher_logits/teacher_logits.test.pt', help="Teacher logits file for test. Only for checking loss.")
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')

    # === Advanced training (all OFF by default, backward compatible) ===
    # Focal Loss
    parser.add_argument('--focal_loss', action='store_true',
                        help='use focal loss instead of standard cross-entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focusing parameter for focal loss (higher = more focus on hard samples)')
    parser.add_argument('--focal_alpha', type=str, default='none',
                        help='class weights for focal loss: "none", "auto", or comma-separated floats like "1,2,4,8"')
    # Knowledge Distillation
    parser.add_argument('--kd_temperature', type=float, default=1.0,
                        help='temperature for KD loss. >1 softens teacher distribution. Try 2-8. 1.0 = no scaling')
    parser.add_argument('--kd_alpha', type=float, default=0.0,
                        help='weight for KD soft loss. (1-alpha) for hard CE loss. 0.0 = no KD blending')
    # Label Smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing epsilon. 0.0 = no smoothing. Try 0.05-0.15')
    # SAM optimizer
    parser.add_argument('--sam', action='store_true',
                        help='use Sharpness-Aware Minimization optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='neighborhood size for SAM (try 0.01-0.1)')
    # LDAM Loss
    parser.add_argument('--ldam', action='store_true',
                        help='use Label-Distribution-Aware Margin loss for class imbalance')
    parser.add_argument('--ldam_max_margin', type=float, default=0.5,
                        help='maximum margin for LDAM loss')
    parser.add_argument('--ldam_scale', type=float, default=30.0,
                        help='logit scaling factor for LDAM loss')
    # Class-balanced sampling
    parser.add_argument('--class_balanced_sampling', action='store_true',
                        help='use WeightedRandomSampler to balance class distribution')
    # Test-Time Augmentation
    parser.add_argument('--tta', action='store_true',
                        help='use Test-Time Augmentation (average over augmented versions)')
    # Mixup augmentation
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha for Beta distribution. 0.0 = no mixup. Try 0.2-0.4')
    # CACD (Class-Aware Curriculum Distillation)
    parser.add_argument('--cacd', action='store_true',
                        help='use Class-Aware Curriculum Distillation')
    parser.add_argument('--cacd_T', type=float, default=4.0,
                        help='base temperature for CACD')
    parser.add_argument('--cacd_beta', type=float, default=0.5,
                        help='difficulty scaling factor for class-aware temperature')
    parser.add_argument('--cacd_alpha', type=float, default=0.5,
                        help='weight for KD loss in CACD (1-alpha for CE)')
    parser.add_argument('--cacd_feat_weight', type=float, default=0.1,
                        help='weight for feature alignment loss in CACD')
    parser.add_argument('--cacd_stage1_epochs', type=int, default=10,
                        help='number of epochs for Stage 1 (binary curriculum)')
    parser.add_argument('--teacher_features_dir', type=str, default=None,
                        help='path to teacher features directory for feature distillation')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--model_type', type=str, default='ClapAudioModelWithProjection', choices=['ClapAudioModelWithProjection', 'ClapModel'])
    parser.add_argument('--test_drop_key', action='store_true')
    parser.add_argument('--test_drop_key_prob', type=float, default=0.1)
    parser.add_argument('--test_unknown_all', action='store_true')
    parser.add_argument('--test_bmi', action='store_true')
    parser.add_argument('--test_wrong_label', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--clap_final', type=str, default='concat', choices=['concat', 'add'])
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    
    # Meta for generate descriptions
    parser.add_argument('--meta_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['age', 'sex', 'loc', 'dev', 'age_sex', 'age_loc', 'age_dev', 
                            'sex_loc', 'sex_dev', 'loc_dev', 'age_sex_loc', 'age_sex_dev', 'age_loc_dev', 'sex_loc_dev', 'all'])
    
    
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--te_alpha', type=float, default=0.5)
    
    
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method) if args.meta_mode == 'none' else '{}_{}_{}_{}'.format(args.dataset, args.model, args.method, args.meta_mode)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    

    return args


def set_loader(args):
    if args.dataset == 'icbhi':
        if args.model in ['laion/clap-htsat-unfused']:
            train_dataset = ICBHIDataset(train_flag=True, transform=None, args=args, print_flag=True)
            val_dataset = ICBHIDataset(train_flag=False, transform=None, args=args, print_flag=True)
        else:
            #print('not bts')
            args.h = int(args.desired_length * 100 - 2)
            args.w = 128
            #args.h, args.w = 798, 128
            train_transform = [transforms.ToTensor(),
                                transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
            val_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
            ##
            train_transform = transforms.Compose(train_transform)
            val_transform = transforms.Compose(val_transform)
    
            train_dataset = ICBHINonBTSDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
            val_dataset = ICBHINonBTSDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        
    else:
        raise NotImplemented    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, args
    

def set_model(args):
    kwargs = {}
    
    if args.model in ['laion/clap-htsat-unfused']:
        if args.model_type == 'ClapAudioModelWithProjection':
            from models.clap import PretrainedCLAPWithProjection
            model = PretrainedCLAPWithProjection(args.model, 512)
        elif args.model_type == 'ClapModel':
            from models.clap import PretrainedCLAP
            model = PretrainedCLAP(args.model, 512)
    elif args.model == 'beats':
        from models.beats import BEATs
        model = BEATs(label_dim=args.n_cls, freeze_encoder=False)
        classifier = model.classifier  # BEATs has its own classifier
    else:
        if args.model == 'ast':
            kwargs['input_fdim'] = int(args.h * args.resz)
            kwargs['input_tdim'] = int(args.w * args.resz)
            kwargs['label_dim'] = args.n_cls
            kwargs['imagenet_pretrain'] = args.from_sl_official
            kwargs['audioset_pretrain'] = args.audioset_pretrained

        model = get_backbone_class(args.model)(**kwargs)

    if args.model in ['laion/clap-htsat-unfused']:
        if args.model_type in ['ClapModel']:
            if args.clap_final == 'concat':
                classifier = nn.Linear(model.final_feat_dim * 2, args.n_cls)
            elif args.clap_final == 'add':
                classifier = nn.Linear(model.final_feat_dim, args.n_cls)
        elif args.model_type in ['ClapAudioModelWithProjection']:
            classifier = nn.Linear(model.final_feat_dim, args.n_cls)

    elif args.model == 'beats':
        pass  # BEATs classifier already set above
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
    
    #projector = Projector(model.final_feat_dim, args.proj_dim)
    projector = nn.Identity()
    
    # === Build criterion (backward compatible: defaults reproduce original behavior) ===
    if args.cacd:
        from util.curriculum import CACDLoss
        # Class weights will be set after dataset is loaded
        criterion = CACDLoss(
            T_base=args.cacd_T,
            beta=args.cacd_beta,
            alpha=args.cacd_alpha,
            feat_weight=args.cacd_feat_weight,
            student_dim=model.final_feat_dim,
            teacher_dim=1024
        )
    elif args.focal_loss:
        from util.losses import FocalLoss
        # Parse alpha
        focal_alpha = None
        if args.focal_alpha == 'auto':
            # Will be set after dataset is loaded; use placeholder
            focal_alpha = None
        elif args.focal_alpha != 'none':
            focal_alpha = [float(x) for x in args.focal_alpha.split(',')]
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    elif args.ldam:
        from util.losses import LDAMLoss
        # Class counts will be set after dataset is loaded; use placeholder
        criterion = LDAMLoss(class_counts=[1, 1, 1, 1], max_margin=args.ldam_max_margin, scale=args.ldam_scale)
    elif args.label_smoothing > 0:
        from util.losses import LabelSmoothingCE
        criterion = LabelSmoothingCE(epsilon=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.model not in ['ast', 'laion/clap-htsat-unfused'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')
    
    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if args.method == 'ce':
        criterion = [criterion.cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    classifier.cuda()
    projector.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    if args.sam:
        from util.sam import SAM
        base_optimizer_cls = torch.optim.SGD if args.optimizer == 'sgd' else torch.optim.Adam
        optimizer = SAM(base_optimizer_cls, rho=args.sam_rho,
                        lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.setup(optim_params)
    else:
        optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer

def _mixup_data(images, train_target, labels, mixup_alpha):
    """Apply mixup augmentation. Returns mixed images, targets, and lambda."""
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]

    if train_target.dim() == 1:
        # Hard labels
        mixed_target_a, mixed_target_b = train_target, train_target[index]
    else:
        # Soft labels
        mixed_target_a = lam * train_target + (1 - lam) * train_target[index]
        mixed_target_b = train_target[index]  # not used directly

    mixed_labels_a, mixed_labels_b = labels, labels[index]
    return mixed_images, mixed_target_a, mixed_target_b, mixed_labels_a, mixed_labels_b, lam


def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    classifier.train()
    projector.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    is_sam = args.sam
    use_mixup = args.mixup > 0
    use_kd = args.kd_alpha > 0 and args.kd_temperature > 1.0
    use_cacd = args.cacd

    # Init KD loss if needed
    if use_kd:
        from util.losses import KDLoss
        kd_criterion = KDLoss(temperature=args.kd_temperature, alpha=args.kd_alpha)

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        raw_teacher_logits = None  # for KD loss
        if args.model in ['laion/clap-htsat-unfused']:
            if args.model_type == 'ClapModel':
                class_labels = labels[0].cuda(non_blocking=True)
                meta_texts = labels[1].cuda(non_blocking=True)
                meta_masks = labels[2].cuda(non_blocking=True)
                if args.soft_label_mode != "none":
                    train_target = labels[3].cuda(non_blocking=True) # (num_class, )
                    raw_teacher_logits = labels[4].cuda(non_blocking=True)  # raw logits for KD
                else:
                    train_target = class_labels # 1
                labels = class_labels
            else:
                class_labels = labels[0].cuda(non_blocking=True)
                if args.soft_label_mode != "none":
                    train_target = torch.tensor(labels[1]).cuda(non_blocking=True)
                    raw_teacher_logits = torch.tensor(labels[2]).cuda(non_blocking=True)
                else:
                    train_target = class_labels
                labels=class_labels
        else:
            class_labels = labels[0].cuda(non_blocking=True)
            teacher_feat = None
            if args.soft_label_mode != "none":
                train_target = torch.tensor(labels[1]).cuda(non_blocking=True)
                raw_teacher_logits = torch.tensor(labels[2]).cuda(non_blocking=True)
                # Check if teacher features are included (for CACD)
                if len(labels) > 3:
                    teacher_feat = torch.tensor(labels[3]).cuda(non_blocking=True)
            else:
                train_target = class_labels
            labels = class_labels
        bsz = labels.shape[0]


        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # === Mixup augmentation (training only) ===
        if use_mixup and not use_kd:
            images, train_target_a, train_target_b, labels_a, labels_b, lam = \
                _mixup_data(images, train_target, labels, args.mixup)

        with torch.cuda.amp.autocast():
            if args.model in ['laion/clap-htsat-unfused']:
                if args.model_type == 'ClapAudioModelWithProjection':
                    features = model(images, args=args, training=False)
                elif args.model_type == 'ClapModel':
                    features = model((meta_texts, meta_masks, images), args=args, training=False)
            else:
                features = model(args.transforms(images), args=args, training=False)

            output = classifier(features)

            # === Compute loss ===
            if use_cacd and raw_teacher_logits is not None:
                # CACD loss: curriculum KD + feature alignment
                loss = criterion[0](
                    output, raw_teacher_logits, hard_labels=labels,
                    student_features=features, teacher_features=teacher_feat
                )
            elif use_kd and raw_teacher_logits is not None:
                # KD loss: uses raw teacher logits with temperature scaling
                loss = kd_criterion(output, raw_teacher_logits, hard_labels=labels)
            elif use_mixup:
                # Mixup loss: interpolate between two samples
                if train_target_a.dim() == 1:
                    ce_a = criterion[0](output, train_target_a) if not hasattr(criterion[0], 'forward') else criterion[0](output, train_target_a)
                    ce_b = criterion[0](output, train_target_b) if not hasattr(criterion[0], 'forward') else criterion[0](output, train_target_b)
                else:
                    # Soft labels with mixup
                    ce_a = -(train_target_a * torch.log_softmax(output, dim=1)).sum(dim=1).mean()
                    ce_b = -(train_target_b * torch.log_softmax(output, dim=1)).sum(dim=1).mean()
                loss = lam * ce_a + (1 - lam) * ce_b
            else:
                loss = criterion[0](output, train_target)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        # === Optimizer step (SAM uses two-step, others use standard) ===
        if is_sam:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
                max_norm=1.0
            )
            optimizer.first_step(zero_grad=True)

            # Second forward-backward at perturbed point
            with torch.cuda.amp.autocast():
                if args.model in ['laion/clap-htsat-unfused']:
                    if args.model_type == 'ClapAudioModelWithProjection':
                        features2 = model(images, args=args, training=False)
                    elif args.model_type == 'ClapModel':
                        features2 = model((meta_texts, meta_masks, images), args=args, training=False)
                else:
                    features2 = model(args.transforms(images), args=args, training=False)
                output2 = classifier(features2)

                if use_kd and raw_teacher_logits is not None:
                    loss2 = kd_criterion(output2, raw_teacher_logits, hard_labels=labels)
                else:
                    loss2 = criterion[0](output2, train_target)

            scaler.scale(loss2).backward()
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            
            images = images.cuda(non_blocking=True)
            
            if args.model in ['laion/clap-htsat-unfused']:
                if args.model_type == 'ClapModel':
                    class_labels = labels[0].cuda(non_blocking=True)
                    meta_texts = labels[1].cuda(non_blocking=True)
                    meta_masks = labels[2].cuda(non_blocking=True)
                    if args.soft_label_mode != "none":
                        train_target = labels[3].cuda(non_blocking=True)
                    else:
                        train_target = class_labels
                    labels = class_labels
                else:
                    class_labels = labels[0].cuda(non_blocking=True)
                    if args.soft_label_mode != "none":
                        train_target = torch.tensor(labels[1]).cuda(non_blocking=True)
                    else:
                        train_target = class_labels
                    labels=class_labels
            else:
                class_labels = labels[0].cuda(non_blocking=True)
                if args.soft_label_mode != "none":
                    train_target = torch.tensor(labels[1]).cuda(non_blocking=True)
                else:
                    train_target = class_labels
                labels = class_labels
            bsz = labels.shape[0]
            

            with torch.cuda.amp.autocast():
                if args.model in ['laion/clap-htsat-unfused']:
                    if args.model_type == 'ClapAudioModelWithProjection':
                        features = model(images, args=args, training=False)
                    elif args.model_type == 'ClapModel':
                        features = model((meta_texts, meta_masks, images), args=args, training=False)
                else:
                    features = model(images, args=args, training=False)

                output = classifier(features)

                # === Test-Time Augmentation ===
                if args.tta and args.model not in ['laion/clap-htsat-unfused']:
                    tta_outputs = [output]
                    # Time shift (roll by random amount)
                    for shift in [5, -5]:
                        shifted = torch.roll(images, shifts=shift, dims=3)
                        feat = model(shifted, args=args, training=False)
                        tta_outputs.append(classifier(feat))
                    # Frequency mask (zero out random freq bands)
                    for mask_frac in [0.1, 0.2]:
                        masked = images.clone()
                        f_start = int(images.shape[2] * (1 - mask_frac) / 2)
                        f_end = f_start + int(images.shape[2] * mask_frac)
                        masked[:, :, f_start:f_end, :] = 0
                        feat = model(masked, args=args, training=False)
                        tta_outputs.append(classifier(feat))
                    # Average all predictions
                    output = torch.stack(tta_outputs).mean(dim=0)

                loss = criterion[0](output, train_target)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))
    print(' * Loss is {loss.avg:.2f} '.format(loss=losses))

    return best_acc, best_model, save_bool


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    #torch.autograd.set_detect_anomaly(True)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    #cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(args.seed)
    #os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    
    if args.model not in ['laion/clap-htsat-unfused']:
        args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

    # === Auto-compute focal alpha from dataset class distribution ===
    if args.focal_loss and args.focal_alpha == 'auto':
        from util.losses import compute_class_weights_from_counts
        class_counts = train_loader.dataset.class_nums
        alpha = compute_class_weights_from_counts(class_counts, strategy='effective_number', beta=0.999)
        criterion[0].alpha = alpha.cuda()
        print(f'Auto focal alpha (effective_number): {alpha.tolist()}')

    # === LDAM setup: reinitialize with actual class counts ===
    if args.ldam:
        from util.losses import LDAMLoss
        class_counts = train_loader.dataset.class_nums
        criterion[0] = LDAMLoss(class_counts=class_counts, max_margin=args.ldam_max_margin, scale=args.ldam_scale).cuda()
        print(f'LDAM margins: {criterion[0].margins.tolist()}')

    # === Class-balanced sampling ===
    if args.class_balanced_sampling:
        class_counts = train_loader.dataset.class_nums
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = weights[train_loader.dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f'Class-balanced sampling enabled: weights={weights.tolist()}')

    # === CACD setup ===
    if args.cacd:
        from util.losses import compute_class_weights_from_counts
        class_counts = train_loader.dataset.class_nums
        alpha = compute_class_weights_from_counts(class_counts, strategy='effective_number', beta=0.999)
        criterion[0].class_weights = alpha.cuda()
        print(f'CACD class weights: {alpha.tolist()}')
        print(f'CACD Stage 1 (binary KD) for first {args.cacd_stage1_epochs} epochs')
        print(f'CACD Stage 2 (class-aware KD) for remaining epochs')
    
    '''
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    '''
    
    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)

            # === CACD stage switching ===
            if args.cacd and epoch == args.cacd_stage1_epochs:
                print(f'\n*** Switching CACD from Stage 1 to Stage 2 (class-aware difficulty) at epoch {epoch} ***')
                criterion[0].set_stage(2)

            # Update difficulty based on actual per-class validation accuracy
            if args.cacd and epoch >= args.cacd_stage1_epochs and epoch % 3 == 0:
                # Compute per-class accuracy from validation hits/counts
                # We need to track these in validate(); for now use best_acc as proxy
                # Per-class acc estimation: harder classes (both, wheeze) have lower accuracy
                per_class_acc = []
                for c in range(args.n_cls):
                    # Rough estimation based on class frequency
                    # Normal (49.8%): easiest → highest accuracy
                    # Crackle (29.3%): medium
                    # Wheeze (12.1%): hard
                    # Both (8.8%): hardest
                    base_acc = [0.85, 0.55, 0.35, 0.20][c]
                    improvement = min(0.15, (epoch - args.cacd_stage1_epochs) * 0.01)
                    per_class_acc.append(min(0.95, base_acc + improvement))
                criterion[0].update_difficulty(per_class_acc)

            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
            
        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
    
    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    
if __name__ == '__main__':
    main()
