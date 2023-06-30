import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from models import get_embedding_layer, create_model
from transformers import GPT2TokenizerFast
from misc.colors import Colors
import csv
from datetime import datetime

import torch.nn as nn

import deepspeed
from datetime import timedelta
os.environ["TOKENIZERS_PARALLELISM"] = "true"


CONTEXT_MAX_LENGTH_FOR_125M = 750 # 전체 1,024에서 라벨(Max 200), 프롬프트(Max 40)을 제외하고 어림잡아 설정

SUMMARY_TOKEN = '[unused0]'
BIG_SEP_TOKEN = '[unused1]'
MID_SEP_TOKEN = '[unused2]'
SMALL_SEP_TOKEN = '[unused3]'
LAST_SEP_TOKEN = '[unused4]'

SUMMARY_TOKENS_IDS = [213, 365, 1291, 316, 518, 283, 213] # '\n요약하면,\n'

global_rank = int(os.getenv('RANK', '0'))
local_rank = 0
world_size = 1


def log(string):
    if global_rank == 0:
        print(string)


def init_distributed(dist_backend='nccl', timeout=timedelta(minutes=60)):
    torch.distributed.init_process_group(backend=dist_backend,
                                        timeout=timeout,
                                        init_method = None)
                                        # init_method='tcp://127.0.0.1:40001',
                                        # world_size=2, 
                                        # rank=1)


def create_model_parallel_group():
    print('create_model_parallel_group()')

    # Call the init process
    init_distributed()
    torch.cuda.set_device(local_rank)
    ranks = [i for i in range(world_size)]
    mp_group = torch.distributed.new_group(ranks)
    return mp_group


class TuningForDownstreamTask(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.device = device
        # load pre-trained model
        self.model, self.tokenizer = create_model(self.args)
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.eos_token = '</s>'
        if world_size > 1:
            mp_group = create_model_parallel_group()
            if self.args.precision == 'fp16':
                self.model = deepspeed.init_inference(self.model, mp_size=world_size, mpu=mp_group, dtype=torch.half, replace_method='auto')
            elif self.args.precision == 'fp32' or self.args.precision == 'mp':
                self.model = deepspeed.init_inference(self.model, mp_size=world_size, mpu=mp_group, dtype=torch.float, replace_method='auto')
            else:
                raise NotImplementedError('fp16/fp32/mp 이외 Precision은 지원하지 않습니다.')
        else:
            if self.args.precision == 'fp16':
                self.model = self.model.half()
            self.model = self.model.to(self.device)

        self.model_max_length = self.model.config.n_positions
        self.max_context_length = self.args.max_context_length


        if self.args.mode == 'finetune':
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    def embed_input(self, queries):
        queries_for_embedding = queries.clone()
        raw_embeds = self.embeddings(queries_for_embedding)

        return raw_embeds

    def get_query_label(self, context, label=None, generate=False):

        context_ids = self.tokenizer(context).input_ids

        if len(context_ids) > self.max_context_length:
            context_ids = context_ids[:self.max_context_length]

        if generate==True:
            return context_ids

        else:
            label_ids = self.tokenizer(label).input_ids + [self.eos_token_id]
            input_ids = context_ids
            label_ids = [-100]*(len(input_ids) - len(label_ids)) + label_ids

            # if len(input_ids) > self.args.max_sequence_length:
            #     label_ids_length = len(label_ids)
            #     oversize = label_ids_length
            #     input_ids = (context_ids[:(self.args.max_sequence_length - oversize)] + label_ids)

            #     log(f' Input ids({len(input_ids)}) is over max seq. length({self.args.max_sequence_length}), Token Oversize: {oversize}')

        return input_ids, label_ids

    def pad_sequence_max(self, queries, padding_value):
        padded_tokens = [query_i + [padding_value] * (self.args.max_sequence_length - len(query_i)) for query_i in queries]

        return torch.LongTensor(padded_tokens).long().to(self.device)


    def pad_sequence_left(self, queries, padding_value):
        queries_len = [len(s) for s in queries]
        max_query_len = max(queries_len)
        padded_tokens = [[padding_value] * (max_query_len - len(query_i)) + query_i for query_i in queries]

        return torch.LongTensor(padded_tokens).long().to(self.device)


    def generate(self, contexts, golds=None):
        queries=[]
        bz = len(contexts)
        for i in range(bz):
            query = self.get_query_label(contexts[i], None, True)
            queries.append(query)

        queries = self.pad_sequence_left(queries, padding_value=self.pad_token_id)
        attention_mask = queries != self.pad_token_id

        outputs_ids = None
        if(self.args.do_sample == True):
            outputs_ids = self.model.generate(input_ids=queries,
                                                attention_mask=attention_mask.to(self.device),
                                                max_length=min(len(queries[0])+self.args.generation_max_length, self.model_max_length),
                                                eos_token_id=self.eos_token_id,
                                                pad_token_id=3,
                                                do_sample=True,
                                                top_k=50,
                                                top_p=0.92,
                                                temperature=0.9)
        else:
            outputs_ids = self.model.generate(input_ids=queries,
                                                attention_mask=attention_mask.to(self.device),
                                                max_length=min(len(queries[0])+self.args.generation_max_length, self.model_max_length),
                                                eos_token_id=self.eos_token_id,
                                                pad_token_id=3,
                                                num_beams=self.args.num_beams)

        predictions = []
        for i, output_id in enumerate(outputs_ids):
            decoded = ''
            decoded = self.tokenizer.decode(output_id[len(queries[i]):], skip_special_tokens=True)
            end_index = decoded.rfind(self.eos_token) if decoded.rfind(self.eos_token) != -1 else None
            predictions.append(decoded[:end_index].strip().replace('\n', '\\n'))

            # if self.args.save_test_inference != None and global_rank == 0:
            #     with open(self.args.save_test_inference, 'a') as f:
            #         wr = csv.writer(f, delimiter="\t")
            #         wr.writerow([golds[i], predictions[i]])

            # 추론이 Null String 인 경우
            if decoded == '' and global_rank == 0:
                print('-------------------------------------------------------')
                print('max_length:', min(len(queries[0])+self.args.generation_max_length, self.model_max_length))
                print('입력 토큰 길이:', len(queries[i]))
                print('추론 토큰 길이:', len(output_id))
                print('추론 결과:', decoded)
                print('-------------------------------------------------------')

        return predictions


    def forward(self, contexts, labels=None, cls_list = None, return_candidates=False):
        batch_size = len(contexts)

        # construct query ids
        queries_ids=[]
        labels_ids=[]
        for i in range(batch_size):
            query, label = self.get_query_label(contexts[i], labels[i] if labels != None else None)

            if self.args.use_pad_sequence_max == True:
                queries_ids.append(query)
                labels_ids.append(label)
            else:
                queries_ids.append(torch.LongTensor(query).squeeze(0))
                labels_ids.append(torch.LongTensor(label).squeeze(0))

        # if self.args.use_pad_sequence_max == True:
        #     queries_ids = self.pad_sequence_max(queries_ids, padding_value=self.pad_token_id)
        #     labels_ids = self.pad_sequence_max(labels_ids, padding_value=-100)
        
        queries_ids = self.pad_sequence_left(queries_ids, padding_value=self.pad_token_id).long().to(self.device)
        labels_ids = self.pad_sequence_left(labels_ids, padding_value=self.pad_token_id).long().to(self.device)

        attention_mask = queries_ids != self.pad_token_id

        outputs = self.model(input_ids=queries_ids,
                            attention_mask=attention_mask.to(self.device),
                            labels=labels_ids.to(self.device) if labels != None else None)
                            # label_smoothing=self.args.label_smoothing)

        # if self.args.focal_usage:
        #     if self.args.focal_type == 'Normal':
        #         NF = Normal_Focalloss(gamma= self.args.focal_gamma, alpha= self.args.focal_alpha, weight=cls_list.to(self.device))
        #         loss = NF(outputs.logits.view(-1, self.args.num_labels), labels_ids.view(-1))
        #         logits = outputs.logits
        #     elif self.args.focal_type == 'Class':
        #         CF = Class_Balanced_FocalLoss(gamma= self.args.focal_gamma, alpha= self.args.focal_alpha, weight=cls_list.to(self.device))
        #         loss = CF(outputs.logits.view(-1, self.args.num_labels), labels_ids.view(-1))
        #         logits = outputs.logits


        # elif self.args.ldam_usage:
        #     LL = LDAMLoss(self.device, cls_num_list=cls_list)
        #     loss = LL(outputs.logits.view(-1, self.args.num_labels), labels_ids.view(-1))
        #     logits = outputs.logits


        # elif self.args.new_cb_usage:
        #     loss = CB_loss(device = self.device, labels = labels_ids.view(-1), \
        #                         logits = outputs.logits.view(-1, self.args.num_labels), samples_per_cls = cls_list, \
        #                         no_of_classes = self.args.num_labels, loss_type = self.args.new_cb_type, beta = 0.9999, gamma = 0.5)
        #     logits = outputs.logits
        # else:
        loss, logits = (outputs.loss, outputs.logits)

        return loss, logits

