import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import numpy as np
import torch.nn.functional as F


def save_task_perf(savepath, results, n_tasks):

    results_array = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i >= j:
                results_array[i, j] = results[i][j]

    np.savetxt(savepath, results_array, fmt='%.2f')


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if type(dataset.N_CLASSES_PER_TASK) == list:
        THIS_TASK_START = int(np.sum(dataset.N_CLASSES_PER_TASK[:k]))
        THIS_TASK_END = int(np.sum(dataset.N_CLASSES_PER_TASK[:k+1]))
    else:
        THIS_TASK_START = k * dataset.N_CLASSES_PER_TASK
        THIS_TASK_END = (k + 1) * dataset.N_CLASSES_PER_TASK

    outputs[:, :THIS_TASK_START] = -float('inf')
    outputs[:, THIS_TASK_END:] = -float('inf')


def evaluate_gt(model, dataset, task='gt', last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                out_feats = model.net.features(inputs, k)
            else:
                out_feats = model.net.features(inputs)

            outout_ae = torch.zeros((out_feats.shape[0], len(dataset.test_loaders), out_feats.shape[-1]),
                                          device=model.device)

            if task == 'mse' or task == 'gt':
                err_ae_1 = torch.zeros((out_feats.shape[0], len(dataset.test_loaders)), device=model.device)
                for i in range(len(dataset.test_loaders)):
                    out_ae_i = model.net.ae[i](out_feats)
                    recon_e = F.mse_loss(out_ae_i, out_feats, reduction='none')
                    err_ae_1[:, i] = torch.mean(recon_e, dim=1)
                    outout_ae[:, i, :] = out_ae_i

                if task == 'mse':
                    indices = torch.argmin(err_ae_1, dim=1)
                    if model.args.use_batch_for_evaluation:
                        tmp = torch.mode(torch.argmin(err_ae_1, dim=1), 0)
                        indices = torch.ones(out_feats.shape[0], device=model.device).long() * tmp[0]
                elif task == 'gt':
                    indices = torch.ones(out_feats.shape[0], device=model.device).long() * k
                mask1 = F.one_hot(indices, len(dataset.test_loaders)).unsqueeze(2)
                mask1 = mask1.expand(-1, -1, out_feats.shape[-1])
                outout_ae = torch.sum(outout_ae * mask1, keepdim=True, dim=1).squeeze()

            outputs = model.net.linear(out_feats * outout_ae)
            _, pred = torch.max(outputs.data , 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    print("Ground Truth Evaluation: ", accs)
    return accs, accs_mask_classes


def evaluate(model: ContinualModel, dataset: ContinualDataset, eval_ema=False, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    curr_model = model.net
    if eval_ema:
        print('setting evaluation model to EMA model')
        curr_model = model.ema_model

    status = curr_model.training
    curr_model.eval()
    # evaluate_gt(model, dataset)
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        acc = []
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                out_feats = curr_model.features(inputs, k)
            else:
                out_feats = curr_model.features(inputs)

            outout_ae = torch.zeros((out_feats.shape[0], len(dataset.test_loaders), out_feats.shape[-1]),
                                          device=model.device)
            err_ae_1 = torch.zeros((out_feats.shape[0], len(dataset.test_loaders)), device=model.device)
            for i in range(len(dataset.test_loaders)):
                out_ae_i = curr_model.ae[i](out_feats)
                recon_e = F.mse_loss(out_ae_i, out_feats, reduction='none')
                err_ae_1[:, i] = torch.mean(recon_e, dim=1)
                outout_ae[:, i, :] = out_ae_i

            if model.args.pretext_task == 'mse':
                indices = torch.argmin(err_ae_1, dim=1)
            elif model.args.pretext_task == 'gt':
                indices = torch.ones(out_feats.shape[0], device=model.device).long() * k
            mask1 = F.one_hot(indices, len(dataset.test_loaders)).unsqueeze(2)
            mask1 = mask1.expand(-1, -1, out_feats.shape[-1])
            outout_ae = torch.sum(outout_ae * mask1, keepdim=True, dim=1).squeeze()

            outputs = curr_model.linear(out_feats * outout_ae)
            _, pred = torch.max(outputs.data , 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            acc.append(k == indices)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        acc_mean = torch.cat(acc).float().mean()
        # print('AE selection test accuracy ', acc_mean)
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    curr_model.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    # Load SSL pre-trained model
    if args.pretrained:
        state_dict = torch.load(args.chkpt)
        state_dict2 = model.net.state_dict()
        for (name, param), (name2, param2) in zip(state_dict.items(), state_dict2.items()):
            if str(name).startswith('f.') and param.shape == param2.shape:
                state_dict2[name2] = param
        model.net.load_state_dict(state_dict2, strict=True)

    if args.freeze_backbone:
        for name, param in model.net.named_parameters():
            if any(name.startswith(layer) for layer in model.exclude_layers_start_with):
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.net.to(model.device)
    for i in range(5):
        model.net.ae[i].to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    lst_ema_models = ['ema_model']
    ema_loggers = {}
    ema_results = {}
    ema_results_mask_classes = {}
    ema_task_perf_paths = {}

    for ema_model in lst_ema_models:
        ema_results[ema_model], ema_results_mask_classes[ema_model] = [], []

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, tb_logger.get_log_dir())
        task_perf_path = os.path.join(tb_logger.get_log_dir(),  'task_performance.txt')

        for ema_model in lst_ema_models:
            if hasattr(model, ema_model):
                print('=' * 50)
                print(f'Creating Logger for {ema_model}')
                print('=' * 50)
                path = os.path.join(tb_logger.get_log_dir(), ema_model)
                if not os.path.exists(path):
                    os.makedirs(path)
                ema_loggers[ema_model] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, path)
                ema_task_perf_paths[ema_model] = os.path.join(path, 'task_performance_{}.txt'.format(ema_model))

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()

    random_results_class, random_results_task = evaluate(model, dataset_copy)
    random_results_class_ema, random_results_task_ema = evaluate(model, dataset_copy, eval_ema=True)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()

        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

            for ema_model in lst_ema_models:
                if hasattr(model, ema_model):
                    ema_accs = evaluate(model, dataset, eval_ema=True)
                    ema_results[ema_model][t - 1] = ema_results[ema_model][t - 1] + ema_accs[0]

                    if dataset.SETTING == 'class-il':
                        ema_results_mask_classes[ema_model][t - 1] = ema_results_mask_classes[ema_model][t - 1] + \
                                                                     ema_accs[1]
        n_epochs = args.n_epochs

        for epoch in range(n_epochs):
            loss_main, loss_aux = 0, 0
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss, loss_rot = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss, loss_rot = model.observe(inputs, labels, not_aug_inputs)

                loss_main += loss
                loss_aux += loss_rot
                progress_bar(i, len(train_loader), epoch, t, loss)

            if args.tensorboard:
                tb_logger.log_loss(loss_main / len(train_loader), n_epochs, epoch, t)
                tb_logger.log_loss_pretext_task(loss_aux / len(train_loader), n_epochs, epoch, t)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)

        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
            csv_logger.log(mean_acc)
            for ema_model in lst_ema_models:
                print('=' * 30)
                print(f'Evaluating {ema_model}')
                print('=' * 30)
                ema_accs = evaluate(model, dataset, eval_ema=True)

                ema_results[ema_model].append(ema_accs[0])
                ema_results_mask_classes[ema_model].append(ema_accs[1])
                ema_mean_acc = np.mean(ema_accs, axis=1)
                print_mean_accuracy(ema_mean_acc, t + 1, dataset.SETTING)
                ema_loggers[ema_model].log(ema_mean_acc)

    if args.tensorboard:
        tb_logger.close()
        csv_logger.add_fwt(results, random_results_class, results_mask_classes, random_results_task)
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        csv_logger.write(vars(args))
        save_task_perf(task_perf_path, results, dataset.N_TASKS)
        # plot_task_performance(tb_logger.get_log_dir(), task_perf_path)

        # Evaluate on EMA model
        for ema_model in lst_ema_models:
            if ema_model in ema_loggers:
                ema_loggers[ema_model].add_fwt(results, random_results_class_ema,
                                               results_mask_classes, random_results_task_ema)
                ema_loggers[ema_model].add_bwt(results, results_mask_classes)
                ema_loggers[ema_model].add_forgetting(results, results_mask_classes)
                ema_loggers[ema_model].write(vars(args), write_ema=True)
                save_task_perf(ema_task_perf_paths[ema_model], ema_results[ema_model], dataset.N_TASKS)
                # plot_task_performance(tb_logger.get_log_dir(), ema_task_perf_paths[ema_model])

    # save checkpoint
    fname = os.path.join(tb_logger.get_log_dir(), 'checkpoint.pth')
    torch.save(model.net.state_dict(), fname)

    fname = os.path.join(tb_logger.get_log_dir(), 'ema_checkpoint.pth')
    torch.save(model.ema_model.state_dict(), fname)
