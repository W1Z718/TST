import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial
from collections import defaultdict

import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

from utils import utils, analysis
from models.loss import l2_reg_loss
from datasets.dataset import ImputationDataset, TransductionDataset, ClassiregressionDataset, collate_unsuperv, collate_superv


logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config['task']

    if task == "imputation":
        return partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
                       masking_ratio=config['masking_ratio'], mode=config['mask_mode'],
                       distribution=config['mask_distribution'], exclude_feats=config['exclude_feats']),\
                        collate_unsuperv, UnsupervisedRunner
    if task == "transduction":
        return partial(TransductionDataset, mask_feats=config['mask_feats'],
                       start_hint=config['start_hint'], end_hint=config['end_hint']), collate_unsuperv, UnsupervisedRunner
    if (task == "classification") or (task == "regression"):
        return ClassiregressionDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # �?动创建目�?

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def fold_evaluate(dataset, model, device, loss_module, target_feats, config, dataset_name):

    allfolds = {'target_feats': target_feats,  # list of len(num_folds), each element: list of target feature integer indices
                'predictions': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
                'targets': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
                'target_masks': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
                'metrics': [],  # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
                'IDs': []}  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):

        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_unsuperv(x, max_len=config['max_seq_len']))

        evaluator = UnsupervisedRunner(model, loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

        logger.info("Evaluating {} set, fold: {}, target features: {}".format(dataset_name, i, tgt_feats))
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(per_batch['metrics'], per_batch['target_masks'])
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds['metrics'].append(metrics_array)
        allfolds['predictions'].append(np.concatenate(per_batch['predictions'], axis=0))
        allfolds['targets'].append(np.concatenate(per_batch['targets'], axis=0))
        allfolds['target_masks'].append(np.concatenate(per_batch['target_masks'], axis=0))
        allfolds['IDs'].append(np.concatenate(per_batch['IDs'], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info("{}:: Mean: {:.3f}, std: {:.3f}".format(metric_name, metrics_mean[m], metrics_std[m]))

    pred_filepath = os.path.join(config['pred_dir'], dataset_name + '_fold_transduction_predictions.pickle')
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, 'wb') as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        batch_metrics = np.stack(metrics[b], axis=1)  # (num_active_in_batch, num_metrics)
        ind = 0
        metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator, epoch_num=None, keep_all=True):
    """Evaluate model."""
    total_samples = 0
    total_batches = 0
    epoch_loss = 0
    per_batch = {'predictions': [], 'targets': [], 'metrics': [], 'IDs': []}
    
    for i, batch in enumerate(evaluator.dataloader):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(evaluator.device)
        padding_masks = padding_masks.to(evaluator.device)  # 0s: ignore
        predictions = evaluator.model(X.to(evaluator.device), padding_masks)
        
        loss = evaluator.loss_module(predictions, targets)  # (batch_size,) loss for each sample
        batch_loss = torch.sum(loss).cpu().item()
        mean_loss = batch_loss / len(loss)  # mean loss (over samples)
        
        per_batch['predictions'].append(predictions.detach().cpu().numpy())
        per_batch['targets'].append(targets.cpu().numpy())
        per_batch['metrics'].append(loss.detach().cpu().numpy())  # 直接存储loss数组
        per_batch['IDs'].append(IDs)
        
        total_samples += len(loss)
        epoch_loss += batch_loss
    
    # 在最后统一处理数组
    if keep_all:
        per_batch['predictions'] = np.concatenate(per_batch['predictions'], axis=0)
        per_batch['targets'] = np.concatenate(per_batch['targets'], axis=0)
        per_batch['IDs'] = np.concatenate(per_batch['IDs'], axis=0)
        per_batch['metrics'] = np.concatenate(per_batch['metrics'], axis=0)  # 直接连接loss数组
    
    epoch_loss = epoch_loss / total_samples
    evaluator.epoch_metrics['epoch'] = epoch_num
    evaluator.epoch_metrics['loss'] = epoch_loss
    
    if keep_all:
        return evaluator.epoch_metrics, per_batch
    else:
        return evaluator.epoch_metrics


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        # 处理 per_batch 字典，�'保所有数组都�?规则的形�?
        save_dict = {}
        for key, value in per_batch.items():
            if isinstance(value, np.ndarray):
                # 如果�?数组，直接保�?
                save_dict[key] = value
            elif isinstance(value, list):
                # 如果�?列表，尝试转�?为数�?
                try:
                    save_dict[key] = np.array(value)
                except:
                    # 如果�?换失败，跳过这个�?
                    logger.warning(f"Skipping {key} in predictions save due to irregular shape")
                    continue
            else:
                # 其他类型直接保存
                save_dict[key] = value

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        np.savez(pred_filepath, **save_dict)

    return aggr_metrics, best_metrics, best_value



def check_progress(epoch):

    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # TODO: for debugging
            # input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
            # if not input_ok:
            #     print("Input problem!")
            #     ipdb.set_trace()
            #
            # utils.check_model(self.model, verbose=False, stop_on_error=True)

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

            if keep_all:
                per_batch['target_masks'].append(target_masks.cpu().numpy())
                per_batch['targets'].append(targets.cpu().numpy())
                per_batch['predictions'].append(predictions.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])
                per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):
        super(SupervisedRunner, self).__init__(model, dataloader, device, loss_module, optimizer, l2_reg, print_interval, console)
        
        # 添加嵌入保存相关的参�?
        self.save_embedding_interval = 1  # 每个epoch都保�?
        self.config = None
        
        if isinstance(loss_module, torch.nn.CrossEntropyLoss):
            self.classification = True
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        else:
            self.classification = False

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model = self.model.train()

        epoch_loss = 0
        total_samples = 0

        for i, batch in enumerate(self.dataloader):
            # 现在batch�?包含3�?元素：features, targets, padding_masks
            features, targets, padding_masks = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)
            
            # Forward pass
            predictions = self.model(features.to(self.device), padding_masks)
            
            # Compute loss
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)

            # Backward pass
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            # Print metrics
            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                self.print_callback(i, metrics, prefix=f'Training Epoch {epoch} ')

            total_samples += len(loss)
            epoch_loss += batch_loss.item()

        # 每隔一定epoch保存嵌入
        if hasattr(self, 'save_embedding_interval') and epoch % self.save_embedding_interval == 0:
            try:
                embeddings_dict = self.model.get_embeddings(self.dataloader, self.device)
                if self.config is not None:
                    save_path = os.path.join(self.config['output_dir'], f'embeddings_epoch_{epoch}.pt')
                    torch.save(embeddings_dict, save_path)
                    print(f"Saved embeddings for epoch {epoch}")
            except Exception as e:
                print(f"Warning: Failed to save embeddings - {str(e)}")

        # Compute epoch metrics
        epoch_loss = epoch_loss / total_samples
        self.epoch_metrics['epoch'] = epoch
        self.epoch_metrics['loss'] = epoch_loss
        
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):
        """Evaluate the model"""
        self.model = self.model.eval()

        epoch_loss = 0
        total_samples = 0
        
        # ���ڴ洢����Ԥ�����ʵ��ǩ
        all_predictions = []
        all_targets = []
        
        for i, batch in enumerate(self.dataloader):
            features, targets, padding_masks = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)
            
            # Forward pass
            predictions = self.model(features.to(self.device), padding_masks)
            
            # ����Ԥ�����ʵ��ǩ
            all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Compute loss
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss

        # �������ָ��
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # ����׼ȷ��
        accuracy = np.mean(all_predictions == all_targets)
        
        # ����ÿ������precision, recall��F1����������
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, 
            all_predictions, 
            average='weighted',
            zero_division=0  # ����û��Ԥ�����������
        )
        
        # ��������ָ��
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss / total_samples
        self.epoch_metrics['accuracy'] = accuracy
        self.epoch_metrics['precision'] = precision
        self.epoch_metrics['recall'] = recall
        self.epoch_metrics['weighted_f1'] = f1
        
        # ��ӡ��ϸָ��
        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Weighted F1: {f1:.4f}")
        
        if keep_all:
            # ����per_batch�ֵ��Ա�����ԭ�������
            per_batch = {
                'predictions': all_predictions,
                'targets': all_targets,
                'metrics': np.array([epoch_loss / total_samples])
            }
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
