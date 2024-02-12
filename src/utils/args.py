'''
 # @ Author: Y. Xiao
 # @ Create Time: 2024-02-12 12:11:55
 # @ Modified by: Y. Xiao
 # @ Modified time: 2024-02-12 12:39:40
 # @ Description: Arguments.
 '''


import os
import random
import torch
import argparse
import warnings
import logging 
import six

import torch.backends.cudnn as cudnn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')


def get_args():
    parser = argparse.ArgumentParser()
    model_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
    model_g.add_arg('positionalEncoding',      bool,   True,      'Whether add positional encoding or not before sent into transformer.')
    model_g.add_arg("encoderLayerNum",         int,    12,        "Number of transformer layers.")
    model_g.add_arg("headNum",                 int,    4,         "Number of attention heads.")
    model_g.add_arg("hiddenSize",             int,    256,       "Hidden size.")
    model_g.add_arg("intermediateSize",       int,    512,       "Intermediate size.")
    model_g.add_arg("hiddenAct",              str,    "gelu",    "Hidden act.")
    model_g.add_arg("hiddenDropoutProb",     float,  0.1,       "Hidden dropout ratio.")
    model_g.add_arg("attentionDropoutProb",  float,  0.1,       "Attention dropout ratio.")
    model_g.add_arg("truncatedNormStd",      float,  0.02,      "Initializer range.")
    model_g.add_arg("vocabSize",              int,    None,      "Size of vocabulary.")
    model_g.add_arg("maxSeqLen",             int,    None,      "Max sequence length.")
    model_g.add_arg("weightSharing",          bool,   True,      "If set, share masked lm weights with node embeddings.")
    model_g.add_arg("initCheckpoint",         str,    None,      "Init checkpoint to resume training from.")
    model_g.add_arg("initPretrainedParams", str,    None,
                    "Init pre-trained params which preforms fine-tuning from. "
                    "If 'init_checkpoint' has been set, this argument wouldn't be valid.")
    model_g.add_arg("checkpoints",             str,    "ckpts",   "Path to save checkpoints.")
    model_g.add_arg('need_structural_contrast',bool,   False,     'Use structural contrast.' )

    train_g = ArgumentGroup(parser, "training", "training options.")
    train_g.add_arg("batch_size",        int,    512,                   "Batch size.")
    train_g.add_arg("epoch",             int,    400,                    "Number of training epochs.")
    train_g.add_arg("learning_rate",     float,  5e-4,                   "Learning rate with warmup.")
    train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",  "scheduler of learning rate.",
                    choices=['linear_warmup_decay', 'noam_decay'])
    train_g.add_arg("warmup_proportion", float,  0.1,                    "Proportion of training steps for lr warmup.")
    train_g.add_arg("weight_decay",      float,  1e-2,                   "Weight decay rate for L2 regularizer.")
    train_g.add_arg("use_fp16",          bool,   False,                  "Whether to use fp16 mixed precision training.")
    train_g.add_arg("loss_scaling",      float,  1000.0,
                    "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
    train_g.add_arg('eval_steps',         int,    5,                      'Interval for validation.')
    train_g.add_arg('seed',              int,    0,                     'seed for torch.')

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps",          int,    1000,    "Step intervals to print loss.")
    log_g.add_arg("verbose",             bool,   False,   "Whether to output verbose log.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
    data_g.add_arg('task',                    str,    None,  'task name.')
    data_g.add_arg("train_file",              str,    None,  "Data for training.")
    data_g.add_arg("valid_file",              str,    None,  "Data for evaluating.")
    data_g.add_arg("test_file",               str,    None,  "Data for testing.")
    data_g.add_arg("ground_truth_path",       str,    None,  "Path to ground truth.")
    data_g.add_arg("vocab_path",              str,    None,  "Path to vocabulary.")
    data_g.add_arg("description_path",        str,    None,  "Path to text description.")
    data_g.add_arg('pretrained_model',        str,    'distilbert-base-uncased',  'path to pretrained model.')
    data_g.add_arg('tokenizer',               str,    'distilbert-base-uncased',  'Type of tokenizer.')
    data_g.add_arg('desc_emb_file',           str,    None,  "Path to description embedding file.")
    data_g.add_arg('token_desc_emb_file',     str,    None,  "Path to token description embedding file.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("cuda_id",                     str,   '0',  "If set, use GPU for training.")
    run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor.")
    run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Iteration intervals to clean up temporary variables.")
    run_type_g.add_arg("do_train",                     bool,   False, "Whether to perform training.")
    run_type_g.add_arg("do_predict",                   bool,   False, "Whether to perform prediction.")

    return parser.parse_args()

# print(args)