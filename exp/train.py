import os

import torch

import _init_paths
import time
import shutil
import pytorch_warmup
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from core.config.cfg import cfg, cfg_name
from core.net.end2end import Model
from core.data.ntuannot import NTUAnnot
from core.data.feeder import Feeder
from core.functions.loss import *
from core.functions.utils import *
from core.functions.function import *




def main():
    logger, final_output_dir = create_logger(root_output_dir='../output', name=cfg['NAME'], phase='train')
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    model = Model(cfg, is_train=True)
    if model.is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.apply(weights_init)
        model.hrnet.init_weights(cfg['MODEL']['PRETRAINED'])

    # copy file
    this_dir = os.path.dirname(__file__)
    backup_dir = os.path.join(final_output_dir, 'backup')
    if not os.path.isdir(backup_dir):
        os.mkdir(backup_dir)
    end2end_file = os.path.join(this_dir, '../core/net', 'end2end' + '.py')
    cfg_file = os.path.join(this_dir, '../core/config', cfg_name)
    feeder_file = os.path.join(this_dir, '../core/data', 'feeder' + '.py')
    function_file = os.path.join(this_dir, '../core/functions', 'function' + '.py')
    train_file = os.path.join(this_dir, 'train' + '.py')
    shutil.copy2(end2end_file, backup_dir)
    shutil.copy2(cfg_file, backup_dir)
    shutil.copy2(feeder_file, backup_dir)
    shutil.copy2(function_file, backup_dir)
    shutil.copy2(train_file, backup_dir)

    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: %d" % pytorch_total_params)

    devices = cfg['GPUS']

    model = torch.nn.DataParallel(model, device_ids=devices).to(devices[0])

    if 'load_pretrained' in cfg:
        model.load_state_dict(
            torch.load(cfg['load_pretrained'])['state_dict'])

    if 'load_hrnet' in cfg:
        pretrained_dict = torch.load(cfg['load_hrnet'])['state_dict']
        model_dict = model.state_dict()
        # 关键在于下面这句，从model_dict中读取key、value时，用if筛选掉不需要的网络层
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if (key.find('hrnet') != -1)}
        print(len(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    criterion_hrnet = None
    # define loss function (criterion) and optimizer
    if cfg['LOSS']['TYPE'] == 'JointsMSELoss':
        criterion_hrnet = JointsMSELoss(
            use_target_weight=cfg['LOSS']['USE_TARGET_WEIGHT']
        ).to(devices[0])
    elif cfg['LOSS']['TYPE'] == 'KLDiscretLoss':
        criterion_hrnet = KLDiscretLoss().to(devices[0])

    criterion_stgcn = nn.CrossEntropyLoss().to(devices[0])

    # data loading
    ntu_annot = NTUAnnot(cfg)
    train_dataset = Feeder(cfg, is_train=True, annot_dataset=ntu_annot)
    valid_dataset = Feeder(cfg, is_train=False, annot_dataset=ntu_annot)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['BATCH_PER_GPU'] * len(cfg['GPUS']),
        shuffle=True,
        num_workers=cfg['WORKERS'],
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['BATCH_PER_GPU'] * len(cfg['GPUS']),
        shuffle=False,
        num_workers=cfg['WORKERS'],
        pin_memory=True
    )

    if 'opt2' in cfg:
        hrnet_params = list(map(id, model.module.hrnet.parameters()))
        other_params = filter(lambda p: id(p) not in hrnet_params, model.parameters())
        opt1 = get_optimizer(cfg['opt1']['type'], model.module.hrnet.parameters(), lr=cfg['opt1']['LR'])
        opt2 = get_optimizer(cfg['opt2']['type'], other_params, lr=cfg['opt2']['LR'])
        lr_seqs = [[], [], []]
    else:
        opt1 = get_optimizer(cfg['opt1']['type'], model.parameters(), lr=cfg['opt1']['LR'])
        opt2 = None
        lr_seqs = [[], []]

    results = [[], [], [], [], []]
    begin_epoch = 0
    end_epoch = cfg['TRAIN']['END_EPOCH']
    best_acc = 0.0
    batch_policy = ['OneCycleLR', 'CyclicLR']

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg['AUTO_RESUME'] and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        results = checkpoint['results']
        lr_seqs = checkpoint['lr_seqs']
        model.load_state_dict(checkpoint['state_dict'])
        opt1.load_state_dict(checkpoint['opt1'] if 'opt1' in checkpoint else checkpoint['optimizer'])
        if 'opt2' in cfg:
            opt2.load_state_dict(checkpoint['opt2'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_policy1 = cfg['opt1']['POLICY'] if 'POLICY' in cfg['opt1'] else 'MultiStepLR'
    if lr_policy1 == 'CosineAnnealingLR':
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=end_epoch, eta_min=0,
            last_epoch=begin_epoch - 1
        )
    elif lr_policy1 == 'MultiStepLR':
        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
            opt1, cfg['opt1']['LR_STEP'], cfg['opt1']['LR_FACTOR'],
            last_epoch=begin_epoch - 1
        )
    elif lr_policy1 == 'OneCycleLR':
        anneal_strategy = cfg['opt1']['anneal_strategy'] if 'anneal_strategy' in cfg['opt1'] else 'cos'
        three_phase = cfg['opt1']['three_phase'] if 'three_phase' in cfg['opt1'] else False
        pct_start = cfg['opt1']['pct_start'] if 'pct_start' in cfg['opt1'] else 0.3
        div_factor = cfg['opt1']['max_lr']/cfg['opt1']['LR']
        final_div_factor = cfg['opt1']['LR'] / cfg['opt1']['min_lr'] if 'min_lr' in cfg['opt1'] else 10000.0
        if cfg['opt1']['type'] == 'adam':
            lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
                opt1, max_lr=cfg['opt1']['max_lr'], steps_per_epoch=len(train_loader), epochs=end_epoch,
                pct_start=pct_start, three_phase=three_phase, anneal_strategy=anneal_strategy, cycle_momentum=False,
                div_factor=div_factor, final_div_factor=final_div_factor,
                last_epoch=begin_epoch * len(train_loader) - 1
            )
        else:
            lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
                opt1, max_lr=cfg['opt1']['max_lr'], steps_per_epoch=len(train_loader), epochs=end_epoch,
                pct_start=pct_start, three_phase=three_phase, anneal_strategy=anneal_strategy,
                cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                div_factor=div_factor, final_div_factor=final_div_factor,
                last_epoch=begin_epoch * len(train_loader) - 1
            )
    # elif lr_policy1 == 'CyclicLR':
    #     lr_scheduler1 = torch.optim.lr_scheduler.CyclicLR(
    #         opt1, max_lr=cfg['opt1']['max_lr'], steps_per_epoch=len(train_loader), epochs=end_epoch,
    #         div_factor=25.0,final_div_factor = 10000.0
    #     )
    else:
        raise

    if 'opt2' in cfg:
        lr_policy2 = cfg['opt2']['POLICY'] if 'POLICY' in cfg['opt2'] else 'CosineAnnealingLR'
        if lr_policy2 == 'CosineAnnealingLR':
            lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=end_epoch, eta_min=0,
                last_epoch=begin_epoch - 1
            )
        elif lr_policy2 == 'MultiStepLR':
            lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
                opt2, cfg['opt2']['LR_STEP'], cfg['opt2']['LR_FACTOR'],
                last_epoch=begin_epoch - 1
            )
        elif lr_policy2 == 'OneCycleLR':
            anneal_strategy = cfg['opt2']['anneal_strategy'] if 'anneal_strategy' in cfg['opt2'] else 'cos'
            three_phase = cfg['opt2']['three_phase'] if 'three_phase' in cfg['opt2'] else False
            pct_start = cfg['opt2']['pct_start'] if 'pct_start' in cfg['opt2'] else 0.3
            div_factor = cfg['opt2']['max_lr'] / cfg['opt2']['LR']
            final_div_factor = cfg['opt2']['LR'] / cfg['opt2']['min_lr'] if 'min_lr' in cfg['opt2'] else 10000.0
            if cfg['opt2']['type'] == 'adam':
                lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
                    opt2, max_lr=cfg['opt2']['max_lr'], steps_per_epoch=len(train_loader), epochs=end_epoch,
                    pct_start=pct_start, three_phase=three_phase, anneal_strategy=anneal_strategy, cycle_momentum=False,
                    div_factor=div_factor, final_div_factor=final_div_factor,
                    last_epoch=begin_epoch * len(train_loader) - 1
                )
            else:
                lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
                    opt2, max_lr=cfg['opt2']['max_lr'], steps_per_epoch=len(train_loader), epochs=end_epoch,
                    pct_start=pct_start, three_phase=three_phase, anneal_strategy=anneal_strategy,
                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                    div_factor=div_factor, final_div_factor=final_div_factor,
                    last_epoch=begin_epoch * len(train_loader) - 1
                )
        else:
            raise

    logger.info("=> GPU {}, workers {}, batchsize {}".format(
        cfg['GPUS'], cfg['WORKERS'], int(len(cfg['GPUS'])*cfg['BATCH_PER_GPU'])))

    if cfg['PURE_VALID']:
        logger.info("=> using mode PURE_VALID, S and valid frequency will be set to 1, no training and "
                    "checkpoint saving will process")
    end = time.time()
    for epoch in range(begin_epoch, end_epoch):
        best_model = False
        train_top1acc = ''
        valid_top1acc = ''
        train_loss1 = ''
        valid_loss1 = ''
        if not cfg['PURE_VALID']:
            if 'opt2' in cfg:
                logger.info(
                    f"=> training phase, current learning rate {opt1.state_dict()['param_groups'][0]['lr']} "
                    f"and {opt2.state_dict()['param_groups'][0]['lr']}")
            else:
                logger.info(f"=> training phase, current learning rate {opt1.state_dict()['param_groups'][0]['lr']}")
            if cfg['MODEL']['COORD_REPRESENTATION'] == 'sa-simdr':
                train_top1acc, train_loss1, lr_seq = \
                    train_sa_simdr(cfg, train_loader, model, criterion_hrnet, criterion_stgcn, opt1, epoch, opt2=opt2,
                                   scheduler1=lr_scheduler1 if lr_policy1 in batch_policy else None,
                                   scheduler2=lr_scheduler2 if 'opt2' in cfg and lr_policy2 in batch_policy else None)
            elif cfg['MODEL']['COORD_REPRESENTATION'] == 'heatmap':
                train_top1acc, train_loss1, lr_seq = \
                    train_heatmap(cfg, train_loader, model, criterion_hrnet, criterion_stgcn, opt1, epoch, opt2=opt2,
                                  scheduler1=lr_scheduler1 if lr_policy1 in batch_policy else None,
                                  scheduler2=lr_scheduler2 if 'opt2' in cfg and lr_policy2 in batch_policy else None)
        if epoch % cfg['VALID_FREQ'] == 0 or epoch >= 2 * cfg['NUM_S'] or cfg['PURE_VALID']:
            logger.info("=> validation phase")
            if cfg['MODEL']['COORD_REPRESENTATION'] == 'sa-simdr':
                valid_top1acc, valid_loss1 = validate_sa_simdr(cfg, valid_loader, valid_dataset, model, criterion_hrnet,
                                                               criterion_stgcn, epoch,
                                                               final_output_dir)
            elif cfg['MODEL']['COORD_REPRESENTATION'] == 'heatmap':
                valid_top1acc, valid_loss1 = validate_heatmap(cfg, valid_loader, valid_dataset, model, criterion_hrnet,
                                                              criterion_stgcn, epoch,
                                                              final_output_dir)

        # perf_indicator = validate_sa_simdr(
        #     cfg, valid_loader, valid_dataset, model, criterion,
        #     final_output_dir, tb_log_dir, writer_dict)
        # elif cfg.MODEL.COORD_REPRESENTATION == 'heatmap':
        #     train_heatmap(cfg, train_loader, model, criterion, optimizer, epoch,
        #         final_output_dir, tb_log_dir, writer_dict)
        #
        #     perf_indicator = validate_heatmap(
        #         cfg, valid_loader, valid_dataset, model, criterion,
        #         final_output_dir, tb_log_dir, writer_dict
        #     )
        if not cfg['PURE_VALID']:
            results[0].append(epoch)
            results[1].append(train_top1acc)
            results[2].append(valid_top1acc)
            results[3].append(train_loss1)
            results[4].append(valid_loss1)
            show_results(results, final_output_dir, filename='results.txt')

            seq_len = lr_seq['len']
            opt1_lr_seq = lr_seq['opt1_lr_seq']
            opt2_lr_seq = lr_seq['opt2_lr_seq']

            for k in range(seq_len):
                if seq_len == 1:
                    lr_seqs[0].append(epoch)
                    lr_seqs[1].append(opt1_lr_seq[k])
                    if 'opt2' in cfg:
                        lr_seqs[2].append(opt2_lr_seq[k])
                else:
                    if k % cfg['PRINT_FREQ'] == 0 or k == len(train_loader) - 1:
                        lr_seqs[0].append(f'{epoch}:{k}')
                        lr_seqs[1].append(opt1_lr_seq[k])
                        if 'opt2' in cfg:
                            lr_seqs[2].append(opt2_lr_seq[k])
            show_results(lr_seqs, final_output_dir, filename='lrseqs.txt')

            top1acc = float(valid_top1acc[:-1]) if valid_top1acc != '' else 0.0
            if top1acc > best_acc:
                best_acc = top1acc
                best_model = True

            logger.info('=> saving checkpoint.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg['NAME'],
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'results': results,
                'lr_seqs': lr_seqs,
                'opt1': opt1.state_dict(),
                'opt2': opt2.state_dict() if 'opt2' in cfg else None
            }, final_output_dir)
            logger.info('checkpoint.pth saved')

            if (epoch + 1) % 10 == 0:
                logger.info('=> saving checkpoint_{}.pth'.format(epoch + 1))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg['NAME'],
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'results': results,
                    'lr_seqs': lr_seqs,
                    'opt1': opt1.state_dict(),
                    'opt2': opt2.state_dict() if 'opt2' in cfg else None
                }, final_output_dir, filename='checkpoint_{}.pth'.format(epoch + 1))
                logger.info('checkpoint_{}.pth saved'.format(epoch + 1))

            if best_model:
                logger.info('=> saving checkpoint_best.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg['NAME'],
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'results': results,
                    'lr_seqs': lr_seqs,
                    'opt1': opt1.state_dict(),
                    'opt2': opt2.state_dict() if 'opt2' in cfg else None
                }, final_output_dir, filename='checkpoint_best.pth')
                logger.info('checkpoint_best.pth saved')

            if lr_policy1 not in batch_policy:
                lr_scheduler1.step()
            if 'opt2' in cfg and lr_policy2 not in batch_policy:
                lr_scheduler2.step()
        logger.info(f'epoch {epoch} using {int((time.time() - end) / 60)} minutes')
        end = time.time()


if __name__ == "__main__":
    main()
