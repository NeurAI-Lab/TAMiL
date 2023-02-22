from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def add_arguments(parser):
    parser.add_argument('--pretext_task', type=str, default='mse',
                        help='TAM selection method during training and testing')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone before training')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained backbone (only backbone)')
    parser.add_argument('--chkpt', type=str, default='',
                        help='Pretrained chkpt location.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Loads pre-trained checkpoint to model and evaluates')
    parser.add_argument('--attention', type=str, default='ae_sigmoid',
                        help='Pretrained chkpt location.')
    parser.add_argument('--use_pairwise_discrepancy_loss', action='store_true',
                        help='Use pairwise max discprepancy loss for latent space of auto-encoder ')
    parser.add_argument('--use_pairwise_loss_after_ae', action='store_true',
                        help='Use pairwise max discprepancy loss after AE space of auto-encoder ')
    parser.add_argument('--pairwise_weight', type=float, default=0.1,
                        help='weight for pairwise discrepancy loss for AE')
    parser.add_argument('--code_dim', type=int, default=64,
                        help='Code dimension for AE (sigmoid)')
    parser.add_argument('--reg_weight', type=float, default=0.1,
                        help='EMA regularization weight')
    parser.add_argument('--ema_update_freq', type=float, default=0.05,
                        help='EMA update frequency')
    parser.add_argument('--ema_alpha', type=float, default=0.999,
                        help='EMA alpha')


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')