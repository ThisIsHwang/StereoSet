import argparse
import torch
from tqdm import tqdm

from transformers import LlamaTokenizer
from nlp import load_dataset
from modeling import LlamaWrapper

def get_perplexity(sentence, wrapper, tokenizer, device, max_length, stride):
    encodings = tokenizer(sentence, return_tensors='pt')
    lls_regular = []
    ppl_regular = None
    if stride <= 0:
        stride = max_length

    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            loss_regular = wrapper.compute_loss(input_ids, labels=target_ids)
            log_likelihood_regular = loss_regular * trg_len

        lls_regular.append(log_likelihood_regular)

        ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)
    return ppl_regular

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=-1,
                        help="The maximum input length to be processed (-1 corresponds to the model's context window)")
    parser.add_argument("--stride", type=int, default=-1,
                        help="If set, for the first --stride tokens no loss is computed")

    args = parser.parse_args()
    print(f"Parameters: {args}")
    tokenizer = LlamaTokenizer.from_pretrained("/home/doubleyyh/models/vicuna-13b-1.1")
    wrapper = LlamaWrapper("/home/doubleyyh/models/vicuna-13b-1.1")
    device = "cuda"

    test = "She is a nice girl."
    encodings = tokenizer(test, return_tensors='pt')

    max_length = (args.max_length if args.max_length > 0 else wrapper._model.config.max_position_embeddings) - args.max_length_pattern

    if args.stride <= 0:
        args.stride = max_length

    lls_regular = []
    ppl_regular = None

    for i in tqdm(range(0, encodings.input_ids.size(1), args.stride)):
        begin_loc = max(i + args.stride - max_length, 0)
        end_loc = min(i + args.stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            loss_regular = wrapper.compute_loss(input_ids, labels=target_ids)
            log_likelihood_regular = loss_regular * trg_len

        lls_regular.append(log_likelihood_regular)

        ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)
