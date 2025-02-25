from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import mul

import time
import logging
import os

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F

from core.functions.inference import get_final_preds
from core.data.utils.transform import flip_back, flip_back_simdr
from core.data.utils.transform import transform_preds
from core.functions.vis import save_debug_images

'''mine'''

logger = logging.getLogger(__name__)


def train_sa_simdr(cfg, train_loader, model, criterion_hrnet, criterion_stgcn, opt1, epoch, opt2=None, scheduler1=None,
                   scheduler2=None):
    multi_fuse = cfg['MODEL']['MULTI_FUSE']
    devices = cfg['GPUS']
    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses_hrnet = AverageMeter()
    losses_stgcn = AverageMeter()
    lr_seq = {
        'len': 1 if scheduler2 is None and scheduler1 is None else len(train_loader),
        'opt1_lr_seq': [],
        'opt2_lr_seq': []
    }
    if lr_seq['len'] == 1:
        lr_seq['opt1_lr_seq'].append(opt1.state_dict()['param_groups'][0]['lr'])
        if 'opt2' in cfg:
            lr_seq['opt2_lr_seq'].append(opt2.state_dict()['param_groups'][0]['lr'])
    if multi_fuse:
        losses_vnet = AverageMeter()
        losses_aggregation = AverageMeter()
        stgcn_acc = AverageMeter()
        vnet_acc = AverageMeter()

    top1_accuracy = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    # total_outputx=None
    # total_outputy =None
    for i, (images, targets, target_weights, action) in enumerate(train_loader):
        '''measure data loading time'''
        data_time.update(time.time() - end)

        if lr_seq['len'] != 1:
            lr_seq['opt1_lr_seq'].append(opt1.state_dict()['param_groups'][0]['lr'])
            if 'opt2' in cfg:
                lr_seq['opt2_lr_seq'].append(opt2.state_dict()['param_groups'][0]['lr'])

        '''compute output'''
        images = images.to(devices[0], non_blocking=True)
        targets = targets.to(devices[0], non_blocking=True)
        target_weights = target_weights.float().to(devices[0], non_blocking=True)
        action = action.long().to(devices[0], non_blocking=True)
        action = action.squeeze(-1)
        if multi_fuse:
            output_hrnet, output_stgcn, output_vnet, output_aggregation = model(images)
        else:
            output_hrnet, output_stgcn = model(images)

        # adjust hrnet_batch = batch * frames
        targets = rearrange(targets, 'batch frames joints xyfeatures -> (batch frames) joints xyfeatures')
        target_weights = rearrange(target_weights, 'batch frames joints features -> (batch frames) joints features')

        # get x and y
        len_simdr_x = int(cfg['MODEL']['IMAGE_SIZE'][0] * cfg['MODEL']['SIMDR_SPLIT_RATIO'])
        len_simdr_y = int(cfg['MODEL']['IMAGE_SIZE'][1] * cfg['MODEL']['SIMDR_SPLIT_RATIO'])
        output_x, output_y = torch.split(output_hrnet, [len_simdr_x, len_simdr_y], dim=2)
        target_x, target_y = torch.split(targets, [len_simdr_x, len_simdr_y], dim=2)

        loss_hrnet = criterion_hrnet(output_x, output_y, target_x, target_y, target_weights)

        if multi_fuse:
            loss_stgcn = criterion_stgcn(output_stgcn, action)
            loss_vnet = criterion_stgcn(output_vnet, action)
            loss_aggregation = criterion_stgcn(output_aggregation, action)
            loss = cfg['LOSS']['RATIO'][0] * loss_hrnet + cfg['LOSS']['RATIO'][1] * loss_stgcn + cfg['LOSS']['RATIO'][
                1] * loss_vnet + cfg['LOSS']['RATIO'][1] * loss_aggregation
        else:
            loss_stgcn = criterion_stgcn(output_stgcn, action)
            loss = cfg['LOSS']['RATIO'][0] * loss_hrnet + cfg['LOSS']['RATIO'][1] * loss_stgcn

        # compute gradient and do update step
        opt1.zero_grad()
        if 'opt2' in cfg:
            opt2.zero_grad()
        loss.backward()
        opt1.step()
        if scheduler1 is not None:
            scheduler1.step()
        if 'opt2' in cfg:
            opt2.step()
            if scheduler2 is not None:
                scheduler2.step()
        for name, param in model.named_parameters():
            if param.grad is None:
                logger.warning(name, 'grad is None')

        # measure accuracy and record loss
        if multi_fuse:
            losses_hrnet.update(loss_hrnet.item(), images.size(0))
            losses_stgcn.update(loss_stgcn.item(), images.size(0))
            losses_vnet.update(loss_vnet.item(), images.size(0))
            losses_aggregation.update(loss_aggregation.item(), images.size(0))

            rank = output_stgcn.detach().argsort(dim=1)
            hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
            acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)

            rank = output_vnet.detach().argsort(dim=1)
            hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
            acc2 = sum(hit_top_1) * 1.0 / len(hit_top_1)

            rank = output_aggregation.detach().argsort(dim=1)
            hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
            acc3 = sum(hit_top_1) * 1.0 / len(hit_top_1)

            stgcn_acc.update(acc1)
            vnet_acc.update(acc2)
            top1_accuracy.update(acc3)
        else:
            losses_hrnet.update(loss_hrnet.item(), images.size(0))
            losses_stgcn.update(loss_stgcn.item(), images.size(0))
            rank = output_stgcn.detach().argsort(dim=1)
            hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
            acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)
            top1_accuracy.update(acc1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        speed.update(images.size(0) / batch_time.val)
        end = time.time()

        if i % cfg['PRINT_FREQ'] == 0 or i == len(train_loader) - 1:
            if multi_fuse:
                msg = 'Train: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.avg:.1f}s  \t' \
                      'Speed {speed:.1f}/s\t' \
                      'Data {data_time.avg:.1f}s  \t' \
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.5f})\t' \
                      'Loss2_1 {loss2_1.val:.4f}({loss2_1.avg:.5f})\t' \
                      'Loss2_2 {loss2_2.val:.4f}({loss2_2.avg:.5f})\t' \
                      'Loss2_3 {loss2_3.val:.4f}({loss2_3.avg:.5f})\t' \
                      'AccSTGCN {staccval:.2f}%({staccavg:.3f}%)  \t' \
                      'AccVisNet {vnaccval:.2f}%({vnaccavg:.3f}%) \t' \
                      'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)'.format(
                    epoch, i, len(train_loader) - 1, batch_time=batch_time, speed=speed.avg, data_time=data_time,
                    loss1=losses_hrnet, loss2_3=losses_aggregation,
                    acc1val=top1_accuracy.val * 100, acc1avg=top1_accuracy.avg * 100,
                    loss2_1=losses_stgcn, loss2_2=losses_vnet,
                    staccval=stgcn_acc.val * 100, staccavg=stgcn_acc.avg * 100,
                    vnaccval=vnet_acc.val * 100, vnaccavg=vnet_acc.avg * 100)
            else:
                msg = 'Train: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.avg:.1f}s  \t' \
                      'Speed {speed:.1f}/s\t' \
                      'Data {data_time.avg:.1f}s  \t' \
                      'Loss1 {loss1.val:.4f}({loss1.avg:.5f})\t' \
                      'Loss2 {loss2.val:.4f}({loss2.avg:.5f})\t' \
                      'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)'.format(
                    epoch, i, len(train_loader) - 1, batch_time=batch_time,
                    speed=speed.avg,
                    data_time=data_time, loss1=losses_hrnet, loss2=losses_stgcn,
                    acc1val=top1_accuracy.val * 100, acc1avg=top1_accuracy.avg * 100)
            logger.info(msg)

    return '{acc1avg:.3f}%'.format(acc1avg=top1_accuracy.avg*100), '{loss1.avg:.5f}'.format(loss1=losses_hrnet), lr_seq


def validate_sa_simdr(cfg, val_loader, val_dataset, model, criterion_hrnet, criterion_stgcn, epoch, output_dir):
    multi_fuse = cfg['MODEL']['MULTI_FUSE']
    devices = cfg['GPUS']
    batch_time = AverageMeter()
    speed = AverageMeter()
    losses_hrnet = AverageMeter()
    losses_stgcn = AverageMeter()
    if multi_fuse:
        losses_vnet = AverageMeter()
        losses_aggregation = AverageMeter()
        stgcn_acc = AverageMeter()
        vnet_acc = AverageMeter()
    top1_accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # total_outputx = None
        # total_outputy = None
        for i, (images, targets, target_weights, action, meta) in enumerate(val_loader):
            # compute output

            images = images.to(devices[0], non_blocking=True)
            targets = targets.to(devices[0], non_blocking=True)
            target_weights = target_weights.float().to(devices[0], non_blocking=True)
            action = action.squeeze(-1).long().to(devices[0], non_blocking=True)
            if multi_fuse:
                output_hrnet, output_stgcn, output_vnet, output_aggregation = model(images)
            else:
                output_hrnet, output_stgcn = model(images)

            # adjust hrnet_batch = batch * frames
            targets = rearrange(targets, 'batch frames joints xyfeatures -> (batch frames) joints xyfeatures')
            target_weights = rearrange(target_weights, 'batch frames joints features -> (batch frames) joints features')

            # get x and y
            len_simdr_x = int(cfg['MODEL']['IMAGE_SIZE'][0] * cfg['MODEL']['SIMDR_SPLIT_RATIO'])
            len_simdr_y = int(cfg['MODEL']['IMAGE_SIZE'][1] * cfg['MODEL']['SIMDR_SPLIT_RATIO'])
            output_x, output_y = torch.split(output_hrnet, [len_simdr_x, len_simdr_y], dim=2)
            target_x, target_y = torch.split(targets, [len_simdr_x, len_simdr_y], dim=2)

            loss_hrnet = criterion_hrnet(output_x, output_y, target_x, target_y, target_weights)
            # print(loss_hrnet)

            if multi_fuse:
                loss_stgcn = criterion_stgcn(output_stgcn, action)
                loss_vnet = criterion_stgcn(output_vnet, action)
                loss_aggregation = criterion_stgcn(output_aggregation, action)
            else:
                loss_stgcn = criterion_stgcn(output_stgcn, action)

            # measure accuracy and record loss
            if multi_fuse:
                losses_hrnet.update(loss_hrnet.item(), images.size(0))
                losses_stgcn.update(loss_stgcn.item(), images.size(0))
                losses_vnet.update(loss_vnet.item(), images.size(0))
                losses_aggregation.update(loss_aggregation.item(), images.size(0))

                rank = output_stgcn.argsort(dim=1)
                hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
                acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)

                rank = output_vnet.argsort(dim=1)
                hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
                acc2 = sum(hit_top_1) * 1.0 / len(hit_top_1)

                rank = output_aggregation.argsort(dim=1)
                hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
                acc3 = sum(hit_top_1) * 1.0 / len(hit_top_1)

                stgcn_acc.update(acc1)
                vnet_acc.update(acc2)
                top1_accuracy.update(acc3)
            else:
                losses_hrnet.update(loss_hrnet.item(), images.size(0))
                losses_stgcn.update(loss_stgcn.item(), images.size(0))
                rank = output_stgcn.argsort(dim=1)
                hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
                acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)
                top1_accuracy.update(acc1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            speed.update(images.size(0) / batch_time.val)
            end = time.time()

            # c is 8 , 2
            # center is 8*32,2
            # c = meta['centers']
            # s = meta['scales']
            # center = torch.zeros((images.size(0) * images.size(1), 2), dtype=torch.float32)
            # scale = torch.zeros((images.size(0) * images.size(1), 2), dtype=torch.float32)
            #
            # for batch_idx in range(images.size(0)):
            #     center[batch_idx * images.size(1):(batch_idx + 1) * images.size(1), :] = c[batch_idx:batch_idx + 1, :]
            #     scale[batch_idx * images.size(1):(batch_idx + 1) * images.size(1), :] = s[batch_idx:batch_idx + 1, :]
            #
            # center = center.numpy()
            # scale = scale.numpy()
            # print(center)
            # print(scale)
            if i % cfg['PRINT_FREQ'] == 0 or i == len(val_loader) - 1 or cfg['PURE_VALID']:
                output_x = F.softmax(output_x, dim=2)
                output_y = F.softmax(output_y, dim=2)
                ''' 
                preds_x and y (4*32, 25, 1) 索引
                output        (4*32, 25, 2) x,y坐标点
                '''
                max_val_x, preds_x = output_x.max(2, keepdim=True)
                max_val_y, preds_y = output_y.max(2, keepdim=True)
                """waiting"""
                # mask = max_val_x > max_val_y
                # max_val_x[mask] = max_val_y[mask]
                # maxvals = max_val_x.cpu().numpy()
                """waiting"""
                output = torch.ones([images.size(0) * images.size(1), preds_x.size(1), 2])
                output[:, :, 0] = torch.squeeze(torch.true_divide(preds_x, cfg['MODEL']['SIMDR_SPLIT_RATIO']))
                output[:, :, 1] = torch.squeeze(torch.true_divide(preds_y, cfg['MODEL']['SIMDR_SPLIT_RATIO']))

                output = output.cpu().numpy()
                preds = output.copy()
                """waiting"""
                # Transform back
                # for j in range(output.shape[0]):
                #     preds[j] = transform_preds(
                #         output[j], center[j], scale[j], [cfg['MODEL']['IMAGE_SIZE'][0], cfg['MODEL']['IMAGE_SIZE'][1]]
                #     )
                # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                # all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # # double check this all_boxes parts
                # all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                # all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                # all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                # all_boxes[idx:idx + num_images, 5] = score
                # # image_path.extend(meta['image'])
                # idx += num_images
                """waiting"""
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(cfg['DEBUG'], images, meta,
                                  rearrange(preds, '(batch frames) joints dim->batch frames joints dim',
                                            batch=images.shape[0]),
                                  prefix)

            if i % cfg['PRINT_FREQ'] == 0 or i == len(val_loader) - 1 or cfg['PURE_VALID']:
                # print(output_stgcn.detach().cpu().numpy())
                # print(action.detach().cpu().numpy())
                if multi_fuse:
                    msg = 'Valid: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.avg:.1f}s  \t' \
                          'Speed {speed:.1f}/s\t\t\t' \
                          'Loss1 {loss1.val:.4f}({loss1.avg:.5f})\t' \
                          'Loss2_1 {loss2_1.val:.4f}({loss2_1.avg:.5f})\t' \
                          'Loss2_2 {loss2_2.val:.4f}({loss2_2.avg:.5f})\t' \
                          'Loss2_3 {loss2_3.val:.4f}({loss2_3.avg:.5f})\t' \
                          'AccSTGCN {staccval:.2f}%({staccavg:.3f}%)\t' \
                          'AccVisNet {vnaccval:.2f}%({vnaccavg:.3f}%)\t' \
                          'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)'.format(
                        epoch, i, len(val_loader) - 1, batch_time=batch_time, speed=speed.avg,
                        loss1=losses_hrnet, loss2_3=losses_aggregation,
                        acc1val=top1_accuracy.val * 100, acc1avg=top1_accuracy.avg * 100,
                        loss2_1=losses_stgcn, loss2_2=losses_vnet,
                        staccval=stgcn_acc.val * 100, staccavg=stgcn_acc.avg * 100,
                        vnaccval=vnet_acc.val * 100, vnaccavg=vnet_acc.avg * 100)
                else:
                    msg = 'Valid: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.avg:.1f}s  \t' \
                          'Speed {speed:.1f}/s\t\t\t' \
                          'Loss1 {loss1.val:.4f}({loss1.avg:.5f})\t' \
                          'Loss2 {loss2.val:.4f}({loss2.avg:.5f})\t' \
                          'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)'.format(
                        epoch, i, len(val_loader) - 1, batch_time=batch_time, speed=speed.avg,
                        loss1=losses_hrnet, loss2=losses_stgcn,
                        acc1val=top1_accuracy.val * 100, acc1avg=top1_accuracy.avg * 100)
                logger.info(msg)

    return '{acc1avg:.3f}%'.format(acc1avg=top1_accuracy.avg * 100), '{loss1.avg:.5f}'.format(loss1=losses_hrnet)

    #     name_values, perf_indicator = val_dataset.evaluate(
    #         config, all_preds, output_dir, all_boxes, image_path,
    #         filenames, imgnums
    #     )

    #     model_name = config.MODEL.NAME
    #     if isinstance(name_values, list):
    #         for name_value in name_values:
    #             _print_name_value(name_value, model_name)
    #     else:
    #         _print_name_value(name_values, model_name)

    #     if writer_dict:
    #         writer = writer_dict['writer']
    #         global_steps = writer_dict['valid_global_steps']
    #         writer.add_scalar(
    #             'valid_loss',
    #             losses.avg,
    #             global_steps
    #         )
    #         if isinstance(name_values, list):
    #             for name_value in name_values:
    #                 writer.add_scalars(
    #                     'valid',
    #                     dict(name_value),
    #                     global_steps
    #                 )
    #         else:
    #             writer.add_scalars(
    #                 'valid',
    #                 dict(name_values),
    #                 global_steps
    #             )
    #         writer_dict['valid_global_steps'] = global_steps + 1

    # return perf_indicator


def train_heatmap(cfg, train_loader, model, criterion_hrnet, criterion_stgcn, opt1, epoch, opt2=None, scheduler1=None,
                  scheduler2=None):
    devices = cfg['GPUS']
    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses_hrnet = AverageMeter()
    losses_stgcn = AverageMeter()
    top1_accuracy = AverageMeter()
    lr_seq = {
        'len': 1 if scheduler2 is None and scheduler1 is None else len(train_loader),
        'opt1_lr_seq': [],
        'opt2_lr_seq': []
    }
    if lr_seq['len'] == 1:
        lr_seq['opt1_lr_seq'].append(opt1.state_dict()['param_groups'][0]['lr'])
        if 'opt2' in cfg:
            lr_seq['opt2_lr_seq'].append(opt2.state_dict()['param_groups'][0]['lr'])
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets, target_weights, action) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if lr_seq['len'] != 1:
            lr_seq['opt1_lr_seq'].append(opt1.state_dict()['param_groups'][0]['lr'])
            if 'opt2' in cfg:
                lr_seq['opt2_lr_seq'].append(opt2.state_dict()['param_groups'][0]['lr'])

        # compute output
        images = images.to(devices[0], non_blocking=True)
        targets = targets.to(devices[0], non_blocking=True)
        target_weights = target_weights.float().to(devices[0], non_blocking=True)
        action = action.long().to(devices[0], non_blocking=True)
        action = action.squeeze(-1)
        output_hrnet, output_stgcn = model(images)

        targets = rearrange(targets, 'batch frames joints h w -> (batch frames) joints h w')
        target_weights = rearrange(target_weights, 'batch frames joints features -> (batch frames) joints features')

        output_hrnet = rearrange(output_hrnet, 'batchandframes joints (h w) -> batchandframes joints h w',
                                 h=cfg['MODEL']['HEATMAP_SIZE'][1])

        # if isinstance(outputs, list):
        #     loss = criterion(outputs[0], target, target_weight)
        #     for output in outputs[1:]:
        #         loss += criterion(output, target, target_weight)
        # else:
        loss_hrnet = criterion_hrnet(output_hrnet, targets, target_weights)

        loss_stgcn = criterion_stgcn(output_stgcn, action)

        loss = cfg['LOSS']['RATIO'][0] * loss_hrnet + cfg['LOSS']['RATIO'][1] * loss_stgcn

        # compute gradient and do update step
        opt1.zero_grad()
        if 'opt2' in cfg:
            opt2.zero_grad()
        loss.backward()
        opt1.step()
        if scheduler1 is not None:
            scheduler1.step()
        if 'opt2' in cfg:
            opt2.step()
            if scheduler2 is not None:
                scheduler2.step()

        # measure accuracy and record loss
        losses_hrnet.update(loss_hrnet.item(), images.size(0))

        losses_stgcn.update(loss_stgcn.item(), images.size(0))

        # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        # acc.update(avg_acc, cnt)

        rank = output_stgcn.argsort(dim=1)
        hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]

        acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)
        top1_accuracy.update(acc1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        speed.update(images.size(0) / batch_time.val)
        end = time.time()

        if i % cfg['PRINT_FREQ'] == 0 or i == len(train_loader) - 1:
            # msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #       'Speed {speed:.1f} samples/s\t' \
            #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #     epoch, i, len(train_loader) - 1, batch_time=batch_time,
            #     speed=input.size(0) / batch_time.val,
            #     data_time=data_time, loss=losses, acc=acc)
            # logger.info(msg)

            # writer = writer_dict['writer']
            # global_steps = writer_dict['train_global_steps']
            # writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            # writer_dict['train_global_steps'] = global_steps + 1

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred * 4, output,
            #                   prefix)
            msg = 'Train: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.1f}s({batch_time.avg:.1f}s)\t' \
                  'Speed {speed:.1f}/s\t' \
                  'Data {data_time.val:.1f}s({data_time.avg:.1f}s)\t' \
                  'Loss1 {loss1.val:.4f}({loss1.avg:.5f})\t' \
                  'Loss2 {loss2.val:.4f}({loss2.avg:.5f})\t' \
                  'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)\t'.format(
                epoch, i, len(train_loader) - 1, batch_time=batch_time,
                speed=speed.avg,
                data_time=data_time, loss1=losses_hrnet, loss2=losses_stgcn,
                acc1val=top1_accuracy.val * 100, acc1avg=top1_accuracy.avg * 100)
            logger.info(msg)
    return '{acc1avg:.3f}%'.format(acc1avg=top1_accuracy.avg * 100), '{loss1.avg:.5f}'.format(loss1=losses_hrnet)


def validate_heatmap(cfg, val_loader, val_dataset, model, criterion_hrnet, criterion_stgcn, epoch, output_dir):
    devices = cfg['GPUS']
    batch_time = AverageMeter()
    losses_hrnet = AverageMeter()
    losses_stgcn = AverageMeter()
    top1_accuracy = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # num_samples = len(val_dataset)
    # all_preds = np.zeros(
    #     (num_samples, config.MODEL.NUM_JOINTS, 3),
    #     dtype=np.float32
    # )
    # all_boxes = np.zeros((num_samples, 6))
    # image_path = []
    # filenames = []
    # imgnums = []
    # idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, targets, target_weights, action, meta) in enumerate(val_loader):
            # compute output
            images = images.to(devices[0], non_blocking=True)
            targets = targets.to(devices[0], non_blocking=True)
            target_weights = target_weights.float().to(devices[0], non_blocking=True)
            action = action.squeeze(-1).long().to(devices[0], non_blocking=True)
            output_hrnet, output_stgcn = model(images)

            targets = rearrange(targets, 'batch frames joints h w -> (batch frames) joints h w')
            target_weights = rearrange(target_weights, 'batch frames joints features -> (batch frames) joints features')

            output_hrnet = rearrange(output_hrnet, 'batchandframes joints (h w) -> batchandframes joints h w',
                                     h=cfg['MODEL']['HEATMAP_SIZE'][1])

            loss_hrnet = criterion_hrnet(output_hrnet, targets, target_weights)

            loss_stgcn = criterion_stgcn(output_stgcn, action)

            losses_hrnet.update(loss_hrnet.item(), images.size(0))
            losses_stgcn.update(loss_stgcn.item(), images.size(0))

            # num_images = input.size(0)
            # # measure accuracy and record loss
            # losses.update(loss.item(), num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            # acc.update(avg_acc, cnt)

            rank = output_stgcn.argsort(dim=1)
            hit_top_1 = [label in rank[i, -1:] for i, label in enumerate(action)]
            acc1 = sum(hit_top_1) * 1.0 / len(hit_top_1)
            top1_accuracy.update(acc1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['centers']
            c = rearrange(c, 'bs frames xy -> (bs frames) xy').numpy()
            s = meta['scales']
            if cfg['DATASET']['CROP_MODE'] == '1-frame+':
                scales = torch.zeros((images.size(0) * images.size(1), s.shape[-1]))
                for batch_idx in range(images.size(0)):
                    scales[batch_idx * images.size(1):(batch_idx + 1) * images.size(1), :] = s[batch_idx:batch_idx + 1,
                                                                                             :]
                s = scales.numpy()
            else:
                s = rearrange(s, 'bs frames xy -> (bs frames) xy').numpy()
            score = np.ones(images.size(0) * images.size(1))

            preds, maxvals = get_final_preds(output_hrnet.clone().cpu().numpy(), c, s, cfg['TEST']['POST_PROCESS'])

            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            # all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # # double check this all_boxes parts
            # all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            # all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            # all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            # all_boxes[idx:idx + num_images, 5] = score
            # image_path.extend(meta['image'])

            # idx += num_images

            if i % cfg['PRINT_FREQ'] == 0 or i == len(val_loader) - 1 or cfg['PURE_VALID']:
                msg = 'Valid: [{0}/{1}]\t' \
                      'Time {batch_time.val:.1f}s({batch_time.avg:.1f}s)\t' \
                      'Loss1 {loss1.val:.4f}({loss1.avg:.5f})\t' \
                      'Loss2 {loss2.val:.4f}({loss2.avg:.5f})\t' \
                      'T1Acc {acc1val:.2f}%({acc1avg:.3f}%)\t'.format(
                    i, len(val_loader) - 1, batch_time=batch_time,
                    loss1=losses_hrnet, loss2=losses_stgcn, acc1val=top1_accuracy.val * 100,
                    acc1avg=top1_accuracy.avg * 100)
                logger.info(msg)
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )

                save_debug_images(cfg['DEBUG'], images, meta,
                                  rearrange(preds, '(batch frames) joints dim->batch frames joints dim',
                                            batch=images.shape[0]) * 4, prefix)
    return '{acc1avg:.3f}%'.format(acc1avg=top1_accuracy.avg * 100), '{loss1.avg:.5f}'.format(loss1=losses_hrnet)

    # name_values, perf_indicator = val_dataset.evaluate(
    #     config, all_preds, output_dir, all_boxes, image_path,
    #     filenames, imgnums
    # )

    # model_name = config.MODEL.NAME
    # if isinstance(name_values, list):
    #     for name_value in name_values:
    #         _print_name_value(name_value, model_name)
    # else:
    #     _print_name_value(name_values, model_name)

    # if writer_dict:
    #     writer = writer_dict['writer']
    #     global_steps = writer_dict['valid_global_steps']
    #     writer.add_scalar(
    #         'valid_loss',
    #         losses.avg,
    #         global_steps
    #     )
    #     writer.add_scalar(
    #         'valid_acc',
    #         acc.avg,
    #         global_steps
    #     )
    #     if isinstance(name_values, list):
    #         for name_value in name_values:
    #             writer.add_scalars(
    #                 'valid',
    #                 dict(name_value),
    #                 global_steps
    #             )
    #     else:
    #         writer.add_scalars(
    #             'valid',
    #             dict(name_values),
    #             global_steps
    #         )
    #     writer_dict['valid_global_steps'] = global_steps + 1


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
