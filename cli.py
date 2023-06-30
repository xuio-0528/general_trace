import numpy as np
import pandas as pd
import torch
import csv
import os, sys
from os.path import join, abspath, dirname
from datetime import datetime
import argparse
import jsonlines
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from misc.evaluate_metric import evaluate_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import torch.nn as nn

from misc.colors import Colors
from misc.reduce_lr_on_plateau_lr_scheduler import ReduceLROnPlateauScheduler
from misc.tri_stage_lr_scheduler import TriStageLRScheduler

from get_gpt import get_gpt
from datasets import load_dataset

from baseline.code_datasets.datasetclass import CodeContestDataset

from modeling import TuningForDownstreamTask

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


global_rank = int(os.getenv('RANK', '0'))
local_rank = 0
world_size = 1

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

SUPPORT_MODELS = ['microsoft/CodeGPT-small-py', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-2.7B', 'Salesforce/codet5-small']



def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_empty_cache", action="store_true")

    parser.add_argument("--model_name", type=str, default='Salesforce/codet5-small', choices=SUPPORT_MODELS)
    parser.add_argument("--tokenizer_path", type=str, default='Salesforce/codet5-small')
    parser.add_argument("--ckpt_pathname", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='./data/codecontest/')
    parser.add_argument("--out_dir", type=str, default='./out/')
    parser.add_argument("--train_basename", type=str, default='codecontest_train.jsonl')
    parser.add_argument("--valid_basename", type=str, default='codecontest_valid.jsonl')
    parser.add_argument("--test_basename", type=str, default='codecontest_test.jsonl')
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_sequence_length", type=int, default=2048, help='학습 입력 Sequence 길이 제한(Context+Label)') # 13B: 1024, 1.3B: 2048
    parser.add_argument("--train_data_size", type=int, default=-1) # Full
    parser.add_argument("--test_data_size", type=int, default=-1) # Full

    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--preprocess_text", type=str2bool, default=True)
    parser.add_argument("--test_original_text", action="store_true")

    parser.add_argument("--use_summary_token", type=str2bool, default=True)
    parser.add_argument("--summary_token_type", type=str, default='Tokens', choices=['Token', 'Tokens'])
    
    parser.add_argument("--task_name", type=str, default='noramalgen')
    

    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--mode", default='finetune', choices=['finetune', 'wte'])

    parser.add_argument("--dataset_type", type=str, default='codecontest',
        choices=[
            'humaneval',
            'apps',
            'codecontest',
        ])
    parser.add_argument("--test_type", type=str, default='valid')
    parser.add_argument("--generation_max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--print_train_metric", action="store_true")
    parser.add_argument("--save_test_inference", type=str, default='./pred/first_gen.jsonl')
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--save_model_path", type=str, default=None)

    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default='mp', choices=['fp16', 'fp32', 'mp'])
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument("--scheduler", type=str, default='CosineScheduleWithWarmUp', choices=['ExponentialLR', 'TriStageLRScheduler', 'ReduceLROnPlateauScheduler', 'CosineScheduleWithWarmUp'])

    # parser.add_argument("--num_labels", type=int, default=2)
    # parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')

    # parser.add_argument('--focal_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--ldam_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--new_cb_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--new_cb_type', type=str, default='focal', help='Normal for Normal Focal loss, Class for Class Balanced Focal Loss')

    # parser.add_argument('--focal_type', type=str, default='Normal', help='Normal for Normal Focal loss, Class for Class Balanced Focal Loss')
    # parser.add_argument('--focal_gamma', type=float, default=2., help='gamma for focal loss')
    # parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha for usage of focal loss')

    args = parser.parse_args()
    log(args)

    if world_size == 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    elif world_size > 1:
        args.n_gpu = 0 if args.no_cuda else world_size

    # post-parsing args
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories

    args.use_pad_sequence_max = True

    args.max_context_length = args.max_sequence_length - 200

    assert args.accumulation_steps > 0
    assert args.max_sequence_length >= args.max_context_length, "Max Sequence 길이가 Max Context 길이보다 길어야 합니다!!!"

    assert not (args.mode == 'wte' and args.precision == 'fp16'), "WTE 모드는 FP16에서 실행되지 않습니다!!!"

    set_seed(args)

    return args

def log(string):
    if global_rank == 0:
        print(string, flush=True)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if world_size > 1:
            self.device = f'cuda:{self.args.local_rank}'
        elif world_size == 1:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = TuningForDownstreamTask(args, self.device)
        self.model.to(self.device)
        print("데이터 읽기 시작")
        if args.eval_only == True:
            self.train_set = None
        else:
            self.train_jsonl = []
            with jsonlines.open(args.data_dir + args.train_basename) as f:
                for line in tqdm(f):
                    self.train_jsonl.append(line)
            self.train_set = CodeContestDataset(pd.DataFrame(self.train_jsonl))
            
            self.valid_jsonl = []            
            with jsonlines.open(args.data_dir + args.valid_basename) as f:
                for line in tqdm(f):
                    self.valid_jsonl.append(line)
            self.valid_set = CodeContestDataset(pd.DataFrame(self.valid_jsonl))
        self.test_jsonl = []
        with jsonlines.open(args.data_dir + args.test_basename) as f:
            for line in tqdm(f):
                self.test_jsonl.append(line)
        self.test_set = CodeContestDataset(pd.DataFrame(self.test_jsonl))
        print("데이터 읽기 종료")
        os.makedirs(self.get_save_path(), exist_ok=True)

        if self.args.eval_only == False:
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True)
        elif self.args.eval_only == True:
            self.train_loader = None
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.args.test_batch_size) 
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)

    def evaluate(self, epoch_idx=0):
        self.model.eval()
        if self.args.test_type == 'valid':
            loader = self.valid_loader
        else:
            loader = self.test_loader

        eval_b_cnt=0
        eval_loss=0
        with torch.no_grad():
            log(f"### START TEST ###")
            torch.cuda.empty_cache()

            predictions = []
            contexts = []
            references = []
            for i, (task_id, context, labels) in enumerate(tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}')):
                predictions += self.model.generate(context, labels)

                _loss,_ = self.model(context, labels)
                contexts += context.replace("\n", "\\n")
                references += labels.replace("\n", "\\n")
                eval_b_cnt+=1
                eval_loss += _loss.item()

            eval_loss = eval_loss/eval_b_cnt
            perplexity = torch.exp(torch.tensor(eval_loss))

            log(f"Test Epoch: {epoch_idx} Loss: {eval_loss} perplexity: {perplexity}\n")

            # Metric
        if self.args.save_test_inference != None and global_rank == 0:
            with open(self.args.save_test_inference, 'a', encoding="utf-8") as f:
                for idx in range(len(references)):
                    f.write({"epoch" : epoch_idx, "task_id" : task_id[idx], "completion" : predictions[idx], "prompt" : contexts[idx]})

        return eval_loss, perplexity

    def get_task_name(self):
        names = [self.args.model_name,
                self.args.mode,
                self.args.dataset_type]
        return "_".join(names)

    def get_save_path(self):
        if self.args.save_model_path != None:
            return join(self.args.save_model_path, self.get_task_name())

        return join(self.args.out_dir, self.args.model_name, self.args.mode, self.get_task_name())

    """ Gradient averaging. """
    def average_gradients(self, model):
        size = float(world_size)
        for name, param in model.named_parameters():
            if param.grad == None:
                continue
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= size

    def get_checkpoint(self, epoch_idx, test_ppl, train_ppl):
        ckpt_name = ''
        if world_size == 1:
            ckpt_name = "epoch_{}_test_{}_train_{}.ckpt".format(epoch_idx, test_ppl, train_ppl)
        elif world_size > 1:
            ckpt_name = "epoch_{}_test_{}_train_{}_{}.ckpt".format(epoch_idx, test_ppl, train_ppl, global_rank)

        embedding = None
        if world_size > 1:
            embedding = self.model.module.state_dict()
        else:
            embedding = self.model.state_dict()
        return {'embedding': embedding,
                'test_ppl': test_ppl,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))

        log("# Checkpoint {} saved.".format(ckpt_name))

    def train(self):
        test_ppl = 100000
        best_ckpt = None
        params=[]

        if self.args.mode == 'finetune':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            params.append({'params': self.model.parameters(), 'lr': self.args.lr})
        elif self.args.mode == 'wte':
            for name, param in self.model.named_parameters():
                if 'wte' in name:
                    param.requires_grad = True
                    params.append({'params': param})
        else:
            raise NotImplementedError('wte/finetune 이외 mode는 지원하지 않습니다.')

        optimizer = None
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            optimizer = AdamW(params, lr=self.args.lr, correct_bias=True)
        else:
            raise NotImplementedError('adam/adamw 이외 optimizer는 지원하지 않습니다.')

        my_lr_scheduler = None

        if self.args.scheduler == 'ExponentialLR':
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        elif self.args.scheduler == 'TriStageLRScheduler':
            total_steps = len(self.train_loader) * self.args.max_epochs // self.args.accumulation_steps
            warmup_steps = 4000 // self.args.accumulation_steps
            hold_steps = 8000 // self.args.accumulation_steps
            my_lr_scheduler = TriStageLRScheduler(
                optimizer,
                init_lr=self.args.lr*1e-4,
                peak_lr=self.args.lr,
                final_lr=self.args.lr*1e-5,
                init_lr_scale=0.01,
                final_lr_scale=0.05,
                warmup_steps=warmup_steps,
                hold_steps=hold_steps,
                decay_steps=total_steps-warmup_steps-hold_steps-200,
                total_steps=total_steps,
            )
        elif self.args.scheduler == 'ReduceLROnPlateauScheduler':
            my_lr_scheduler = ReduceLROnPlateauScheduler(
                optimizer,
                lr=self.args.lr)
        elif self.args.scheduler == 'CosineScheduleWithWarmUp':
            train_steps = len(self.train_loader) * self.args.max_epochs // self.args.accumulation_steps
            warmup_steps = int(train_steps * 0.1)
            my_lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
        else:
            raise NotImplementedError('ExponentialLR/TriStageLRScheduler/ReduceLROnPlateauScheduler 이외 scheduler는 지원하지 않습니다.')

        for epoch_idx in range(1, self.args.max_epochs+1):
            total_loss=0
            train_b_cnt=0

            log("### START TRAIN ###")
            torch.cuda.empty_cache()

            self.model.train()
            scaler = GradScaler()
            if world_size > 1:
                torch.distributed.barrier()
            for steps, batch in enumerate(tqdm(self.train_loader, bar_format='{l_bar}{bar:10}{r_bar}')):
                if world_size > 16: # Multi-node
                    if steps % 10 == 0:
                        log(f'{steps}/{len(self.train_loader)} @ {datetime.now()}')
                elif self.args.precision == 'mp': # mixed_precision
                    with autocast():
                        input_des = batch[1]
                        label = batch[2]
                        loss, _ = self.model(input_des, label)
                        loss = loss / self.args.accumulation_steps # Gradient Accumulation 적용
                        total_loss += loss.item()
                    scaler.scale(loss).backward()
                else:
                    loss, _ = self.model(batch[1], batch[2])
                    loss = loss / self.args.accumulation_steps # Gradient Accumulation 적용
                    total_loss += loss.item()
                    loss.backward()

                train_b_cnt+=1

                if world_size > 1:
                    if self.args.mode == 'wte':
                        self.average_gradients(self.model.module.transformer.wte)
                    elif self.args.mode == 'finetune':
                        self.average_gradients(self.model.module)

                if (steps+1) % self.args.accumulation_steps == 0:
                    if self.args.use_empty_cache == True:
                        torch.cuda.empty_cache()
                    if self.args.precision == 'mp': # mixed_precision
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if self.args.use_empty_cache == True:
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()

                    if self.args.scheduler == 'TriStageLRScheduler' or self.args.scheduler == 'CosineScheduleWithWarmUp':
                        my_lr_scheduler.step()
                    elif self.args.scheduler == 'ReduceLROnPlateauScheduler' and train_b_cnt != 0:
                        my_lr_scheduler.step(total_loss/train_b_cnt) # Train Loss

                if self.args.print_train_metric and steps%10==0:
                    if self.args.scheduler == 'CosineScheduleWithWarmUp':
                        log(f"Train LR: {my_lr_scheduler.get_last_lr()[0]:.10f} Epoch {epoch_idx} Step: {steps} Loss: {total_loss/train_b_cnt:.5f} perplexity: {torch.exp(torch.tensor(total_loss/train_b_cnt)):.5f}")
                    else:
                        log(f"Train LR: {my_lr_scheduler.get_last_lr():.6f} Epoch {epoch_idx} Step: {steps} Loss: {total_loss/train_b_cnt:.5f} perplexity: {torch.exp(torch.tensor(total_loss/train_b_cnt)):.5f}")

            if self.args.scheduler == 'ExponentialLR':
                my_lr_scheduler.step()

            if train_b_cnt != 0:
                total_loss = total_loss/train_b_cnt
            train_ppl = torch.exp(torch.tensor(total_loss))

            log("Train LR: {} Epoch {} Loss: {} perplexity: {}".format(my_lr_scheduler.get_last_lr(),epoch_idx, total_loss, train_ppl))

            # if self.args.save_test_inference != None and global_rank == 0:
            #     with open(self.args.save_test_inference, 'a') as f:
            #         wr = csv.writer(f, delimiter="\t")
            #         wr.writerow(['------------------------------', '------------------------------'])

            test_ppl = None
            if self.args.no_eval != True:
                _, test_ppl = self.evaluate(epoch_idx)

            best_ckpt = self.get_checkpoint(epoch_idx, test_ppl, train_ppl)
            if self.args.save_model == True:
                self.save(best_ckpt)

            torch.cuda.empty_cache()
        return best_ckpt

def main():
    args = construct_generation_args()

    if args.save_test_inference != None and global_rank == 0:
        with open(args.save_test_inference, 'w') as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(['정답', '추론'])

    trainer = Trainer(args)

    if args.eval_only == True:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == '__main__':
    main()