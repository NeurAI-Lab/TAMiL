# Evaluation file testing trained 'expert gate' models
import torch
from utils.training import mask_classes
import torch.nn.functional as F


def load_checkpoint(model, args):
    try:
        state_dict = torch.load(args.chkpt)
        model.net.load_state_dict(state_dict, strict=True)
    except:
        raise("Error")
    model.net.to(model.device)
    model.net.eval()


def evaluate_accuracy(model, dataset, last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()

    for t in range(dataset.N_TASKS):
        train_loader, _ = dataset.get_data_loaders()

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
                out_feats = model.net.features(inputs, k)
            else:
                out_feats = model.net.features(inputs)

            outout_ae = torch.zeros((out_feats.shape[0], len(dataset.test_loaders), out_feats.shape[-1]),
                                    device=model.device)
            err_ae_1 = torch.zeros((out_feats.shape[0], len(dataset.test_loaders)), device=model.device)
            for i in range(len(dataset.test_loaders)):
                out_ae_i = model.net.ae[i](out_feats)
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

            outputs = model.net.linear(out_feats * outout_ae)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            acc.append(k == indices)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        acc_mean = torch.cat(acc).float().mean()
        print('AE selection test accuracy ', acc_mean)
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    print(accs, ' == ', sum(accs) / len(accs))
    print(accs_mask_classes, ' == ', sum(accs_mask_classes) / len(accs_mask_classes))
    return accs, accs_mask_classes