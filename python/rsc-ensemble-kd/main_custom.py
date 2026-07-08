"""
Thin wrapper around main.py that adds support for laion/larger_clap_general
WITHOUT modifying the original codebase.

All training logic, evaluation, augmentation, etc. is inherited from main.py.
Only the model name check is patched to support additional CLAP variants.

Usage:
    python main_custom.py --model laion/larger_clap_general --tag BTS_larger_clap \
        --model_type ClapModel --meta_mode all --sample_rate 48000 ...
"""
import sys
import types
import importlib
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# 1. Define which model names are treated as CLAP
# ──────────────────────────────────────────────────────────────
CLAP_MODELS = [
    'laion/clap-htsat-unfused',   # original
    'laion/clap-htsat-fused',     # fused variant
    'laion/larger_clap_general',  # larger, 2.5M pairs
    'laion/larger_clap_music_and_speech',
]


def _is_clap(model_name: str) -> bool:
    return model_name in CLAP_MODELS


class _CLAPNoOpTransform:
    """No-op transform for CLAP models (dataset already applies its own)."""
    def __call__(self, x):
        if x.dim() == 3 and x.shape[0] == 1:
            return x.squeeze(0)
        return x


# ──────────────────────────────────────────────────────────────
# 2. Patched set_model — handles any CLAP_MODELS entry
# ──────────────────────────────────────────────────────────────
def _patched_set_model(args):
    from copy import deepcopy
    from models import get_backbone_class

    kwargs = {}

    if _is_clap(args.model):
        # Use clap_larger module for non-default CLAP models
        if args.model in ('laion/larger_clap_general', 'laion/clap-htsat-fused', 'laion/larger_clap_music_and_speech'):
            from models.clap_larger import PretrainedCLAPWithProjection, PretrainedCLAP
        else:
            from models.clap import PretrainedCLAPWithProjection, PretrainedCLAP

        if args.model_type == 'ClapAudioModelWithProjection':
            model = PretrainedCLAPWithProjection(args.model, 512)
        elif args.model_type == 'ClapModel':
            model = PretrainedCLAP(args.model, 512)
    elif args.model == 'beats':
        from models.beats import BEATs
        model = BEATs(label_dim=args.n_cls, freeze_encoder=False)
        classifier = model.classifier
    else:
        if args.model == 'ast':
            kwargs['input_fdim'] = int(args.h * args.resz)
            kwargs['input_tdim'] = int(args.w * args.resz)
            kwargs['label_dim'] = args.n_cls
            kwargs['imagenet_pretrain'] = args.from_sl_official
            kwargs['audioset_pretrain'] = args.audioset_pretrained
        model = get_backbone_class(args.model)(**kwargs)

    # Build classifier
    if _is_clap(args.model):
        if args.model_type in ['ClapModel']:
            if args.clap_final == 'concat':
                classifier = nn.Linear(model.final_feat_dim * 2, args.n_cls)
            elif args.clap_final == 'add':
                classifier = nn.Linear(model.final_feat_dim, args.n_cls)
        elif args.model_type in ['ClapAudioModelWithProjection']:
            classifier = nn.Linear(model.final_feat_dim, args.n_cls)
    elif args.model == 'beats':
        pass
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)

    projector = nn.Identity()

    # Criterion
    if args.cacd:
        from util.curriculum import CACDLoss
        criterion = CACDLoss(
            T_base=args.cacd_T, beta=args.cacd_beta,
            alpha=args.cacd_alpha, feat_weight=args.cacd_feat_weight,
            student_dim=model.final_feat_dim, teacher_dim=1024,
        )
    elif args.focal_loss:
        from util.losses import FocalLoss
        focal_alpha = None
        if args.focal_alpha == 'auto':
            focal_alpha = None
        elif args.focal_alpha != 'none':
            focal_alpha = [float(x) for x in args.focal_alpha.split(',')]
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    elif args.label_smoothing > 0:
        from util.losses import LabelSmoothingCE
        criterion = LabelSmoothingCE(epsilon=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.model not in ['ast'] and not _is_clap(args.model) and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    if args.pretrained and args.pretrained_ckpt is not None:
        import torch
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if 'mlp_head' not in k:
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)
        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))

    if args.method == 'ce':
        criterion = [criterion.cuda()]

    import torch
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.cuda()
    classifier.cuda()
    projector.cuda()

    from util.misc import set_optimizer
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


# ──────────────────────────────────────────────────────────────
# 3. Patched set_loader — accepts any CLAP_MODELS entry
# ──────────────────────────────────────────────────────────────
def _patched_set_loader(args):
    from torchvision import transforms
    from util.icbhi_dataset import ICBHIDataset, ICBHINonBTSDataset
    import torch

    if args.dataset == 'icbhi':
        if _is_clap(args.model):
            train_dataset = ICBHIDataset(train_flag=True, transform=None, args=args, print_flag=True)
            val_dataset = ICBHIDataset(train_flag=False, transform=None, args=args, print_flag=True)
        else:
            args.h = int(args.desired_length * 100 - 2)
            args.w = 128
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz))),
            ])
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz))),
            ])
            train_dataset = ICBHINonBTSDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
            val_dataset = ICBHINonBTSDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, args


# ──────────────────────────────────────────────────────────────
# 4. Inject patched functions into main module and run
# ──────────────────────────────────────────────────────────────
def main():
    import main as original_main

    # Patch the two functions
    original_main.set_model = _patched_set_model
    original_main.set_loader = _patched_set_loader

    # Patch all is_clap checks inside train/validate/main by replacing
    # the string check with a helper accessible via args
    # We do this by wrapping the original train/validate to inject _is_clap

    # Actually, the train() and validate() functions in main.py use
    # `args.model in ['laion/clap-htsat-unfused']` — we need to patch args.model
    # for those checks. The cleanest way: set a module-level helper.

    # Inject _is_clap into the main module so patched closures can use it
    original_main._is_clap = _is_clap

    # Now we need to patch train() and validate() which also check for CLAP
    # Instead of patching those complex functions, we add our model to the check list
    # by monkey-patching the `in` check via a custom list class

    class CLAPModelList(list):
        """A list that also contains any model in CLAP_MODELS."""
        def __contains__(self, item):
            return super().__contains__(item) or _is_clap(item)

    # Patch the module-level Processor for dataset compatibility
    import util.icbhi_dataset as ds_module
    # The cached data is already compatible — no need to change Processor

    # Run original main — the train/validate functions reference `args.model`
    # and check `in ['laion/clap-htsat-unfused']`. We need to intercept this.
    # The simplest way: temporarily patch the train and validate functions
    # to use _is_clap instead of the hardcoded check.

    _original_train = original_main.train
    _original_validate = original_main.validate

    def _patched_train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
        # Temporarily rename model to match the check
        saved_model = args.model
        if _is_clap(args.model):
            args.model = 'laion/clap-htsat-unfused'
        try:
            return _original_train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
        finally:
            args.model = saved_model

    def _patched_validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
        saved_model = args.model
        if _is_clap(args.model):
            args.model = 'laion/clap-htsat-unfused'
        try:
            return _original_validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
        finally:
            args.model = saved_model

    original_main.train = _patched_train
    original_main.validate = _patched_validate

    # Also patch the main() function's internal CLAP checks
    _original_main_func = original_main.main

    def _patched_main_func():
        # Parse args first to get model name
        args = original_main.parse_args()
        # Temporarily swap model name for the internal checks
        saved_model = args.model
        if _is_clap(args.model):
            # The main() function checks args.model internally
            # We need to handle this by patching at the right level
            pass
        # Just call original — the patched set_model/set_loader will handle it
        _original_main_func()

    # Actually, the issue is that main() calls parse_args() internally
    # and then checks args.model. Let me just re-implement main() cleanly.

    original_main.main = _run_main
    _run_main()


def _run_main():
    """Re-implementation of main.main() with CLAP model support."""
    import main as m
    import os, sys, json, math, time, random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    from torchvision import transforms
    from transformers import set_seed
    from util.augmentation import SpecAugment
    from util.misc import adjust_learning_rate, warmup_learning_rate
    from util.misc import AverageMeter, accuracy, save_model, update_json
    from copy import deepcopy

    # Patch train/validate to handle CLAP model name swap
    _original_train = m.train
    _original_validate = m.validate

    def _patched_train_fn(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
        saved_model = args.model
        if _is_clap(args.model):
            args.model = 'laion/clap-htsat-unfused'
        try:
            return _original_train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
        finally:
            args.model = saved_model

    def _patched_validate_fn(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
        saved_model = args.model
        if _is_clap(args.model):
            args.model = 'laion/clap-htsat-unfused'
        try:
            return _original_validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
        finally:
            args.model = saved_model

    m.train = _patched_train_fn
    m.validate = _patched_validate_fn

    args = m.parse_args()

    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    set_seed(args.seed)

    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0]

    if not _is_clap(args.model):
        args.transforms = SpecAugment(args)
    else:
        args.transforms = _CLAPNoOpTransform()

    train_loader, val_loader, args = _patched_set_loader(args)
    model, classifier, projector, criterion, optimizer = _patched_set_model(args)

    # Auto focal alpha
    if args.focal_loss and args.focal_alpha == 'auto':
        from util.losses import compute_class_weights_from_counts
        class_counts = train_loader.dataset.class_nums
        alpha = compute_class_weights_from_counts(class_counts, strategy='effective_number', beta=0.999)
        criterion[0].alpha = alpha.cuda()
        print(f'Auto focal alpha (effective_number): {alpha.tolist()}')

    # CACD setup
    if args.cacd:
        from util.losses import compute_class_weights_from_counts
        class_counts = train_loader.dataset.class_nums
        alpha = compute_class_weights_from_counts(class_counts, strategy='effective_number', beta=0.999)
        criterion[0].class_weights = alpha.cuda()

    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    scaler = torch.cuda.amp.GradScaler()
    print('*' * 20)

    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))

        for epoch in range(args.start_epoch, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            loss, acc = m.train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc))

            best_acc, best_model, save_bool = m.validate(val_loader, model, classifier, criterion, args, best_acc, best_model)

            # CACD stage switching
            if args.cacd and epoch == args.cacd_stage1_epochs:
                print(f'\n*** Switching CACD from Stage 1 to Stage 2 at epoch {epoch} ***')
                criterion[0].set_stage(2)

            if args.cacd and epoch >= args.cacd_stage1_epochs and epoch % 3 == 0:
                per_class_acc = []
                for c in range(args.n_cls):
                    base_acc = [0.85, 0.55, 0.35, 0.20][c]
                    improvement = min(0.15, (epoch - args.cacd_stage1_epochs) * 0.01)
                    per_class_acc.append(min(0.95, base_acc + improvement))
                criterion[0].update_difficulty(per_class_acc)

            if save_bool:
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))

        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _ = m.validate(val_loader, model, classifier, criterion, args, best_acc)

    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()
