import glob
import os
import copy
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from ltr.ops.iou3d_nms import iou3d_nms_utils
import numpy as np

from ltr.models import load_data_to_gpu
from tools.eval_utils.track_eval_metrics import Success_torch, Precision_torch
from ltr.utils import tracklet3d_kitti
from ltr.utils import common_utils, commu_utils
import time

def train_one_epoch(model, optimizer, train_loader, model_func, epoch, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    epoch_loss = 0
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        epoch_loss += loss.item()
        loss.backward()

        # detr
        if optim_cfg.GRAD_NORM_CLIP > 0:
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                    
    epoch_loss /= total_it_each_epoch         
    if tb_log is not None:
        tb_log.add_scalar('train/epoch_loss', epoch_loss, epoch)

    if rank == 0:
        pbar.close()
    return accumulated_iter

def val_one_epoch(model, val_loader, optim_cfg, epoch, tb_log=None):
    dataset = val_loader.dataset
    first_index = dataset.first_frame_index
    model.eval()
    Success_main = Success_torch()
    Precision_main = Precision_torch()

    try_false_time = 0
    pbar = tqdm.tqdm(total=len(first_index)-1, leave=False, desc='tracklets', dynamic_ncols=True)
    fps = []
    center_error = []
    for f_index in range(len(first_index)-1):
        st = first_index[f_index]
        if f_index == len(first_index) - 2:
            ov = first_index[f_index+1] + 1
        else:
            ov = first_index[f_index+1]

        length = ov - st - 1
        if length > 0:
            previou_box = None
            for index in range(st+1, ov):
                data = dataset[index]

                if optim_cfg.MODEL_TYPE == 'MM':
                    if index == st+1:
                        previou_box = data['template_gt_box'].reshape(7)
                        first_point = data['or_template_points']
                        Success_main.add_overlap(torch.ones(1).cuda())
                        Precision_main.add_accuracy(torch.zeros(1).cuda())

                    batch_dict = dataset.collate_batch([data])
                    template_voxels = batch_dict['template_voxels']
                    search_voxels = batch_dict['search_voxels']

                    load_data_to_gpu(batch_dict)
                    gt_box = batch_dict['gt_boxes'].view(-1)[:7]

                    try:
                        with torch.no_grad():
                            torch.cuda.synchronize()
                            start = time.time()

                            pred_box, motion_pred_box = model(batch_dict)
                            center_error.append(np.linalg.norm(motion_pred_box[:2] - gt_box.cpu().numpy()[:2], axis=0, keepdims=True))

                            torch.cuda.synchronize()
                            end = time.time()
                            fps.append(end - start)
                    except BaseException:
                        try_false_time += 1
                        pred_box = torch.from_numpy(previou_box).float().cuda()

                    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_box.view(1,-1), gt_box.view(1,-1)).squeeze()
                    accuracy = torch.norm(pred_box[:3] - gt_box[:3])
                    Success_main.add_overlap(iou3d)
                    Precision_main.add_accuracy(accuracy)

                    dataset.set_first_points(first_point)
                    dataset.add_refer_box(pred_box.cpu().numpy())
                    previou_box = pred_box.cpu().numpy()

            dataset.reset_all()
        pbar.update()
    pbar.close()
    avs = Success_main.average.item()
    avp = Precision_main.average.item()
    print('')
    print('Success: ', avs, ', Precision: ', avp)
    print('FPS: ', len(fps) / sum(fps))
    print('try_false_time: ', try_false_time)
    print('Avg Center Error: ', sum(center_error) / len(center_error))
    print('')
    if tb_log is not None:
        tb_log.add_scalar('val/Success', avs, epoch)
        tb_log.add_scalar('val/Precision', avp, epoch)
        tb_log.add_scalar('val/Try_false_times', try_false_time, epoch)
        tb_log.add_scalar('val/FPS', len(fps) / sum(fps), epoch)
        tb_log.add_scalar('val/Center_Error', sum(center_error) / len(center_error), epoch)
    return avs, avp

def train_model(model, optimizer, train_loader, val_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        total_it_val_each_epoch = len(val_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
            total_it_val_each_epoch = len(val_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        val_dataloader_iter = iter(val_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # ================ train model ================#
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                epoch=cur_epoch,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            if tb_log is not None:
                tb_log.add_scalar('meta_data/epoch_lr', cur_lr, cur_epoch)

            lr_scheduler.step(accumulated_iter)
            '''
            val_epoch_node = 25
            if cur_epoch > val_epoch_node:
                success, precision = val_one_epoch(model, val_loader, optim_cfg, epoch=cur_epoch, tb_log=tb_log)
            '''
            #================ save trained model ================#
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import ltr
        version = 'ltr+' + ltr.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
