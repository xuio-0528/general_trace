import os
import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from transformers import RobertaTokenizer, T5ForConditionalGeneration

world_size = int(os.getenv('WORLD_SIZE', '1'))
global_rank = int(os.getenv('RANK', '0'))


def log(string):
    if global_rank == 0:
        print(string)


def create_model(args):
    MODEL_CLASS, tokenizer = get_model_and_tokenizer_class(args)

    model = MODEL_CLASS.from_pretrained(args.model_name)

    if args.eval_only == True:
        assert args.ckpt_pathname != None, 'Evaluate할 ckpt 파일을 지정해 주세요!!!'

        ckpt = torch.load(args.ckpt_pathname, map_location='cpu')
        model.load_state_dict(ckpt)
    elif args.ckpt_pathname != None:
        ckpt = torch.load(args.ckpt_pathname, map_location='cpu')
        model.load_state_dict(ckpt)

        log(f'Fine-tuned Model loaded: {args.ckpt_pathname}')

    return model, tokenizer


def get_model_and_tokenizer_class(args):

    return T5ForConditionalGeneration, RobertaTokenizer.from_pretrained(args.tokenizer_path)
    # else:
    #     raise NotImplementedError('summarization/classification 이외 downstream_task는 지원하지 않습니다.')


def get_embedding_layer(args, model):
    if world_size > 1:
        embeddings = model.module.get_input_embeddings()
    else:
        embeddings = model.get_input_embeddings()

    return embeddings
