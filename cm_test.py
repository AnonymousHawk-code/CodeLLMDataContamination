from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import roc_auc_score
import zlib
import numpy as np


from datasets import load_dataset, load_from_disk

@torch.no_grad()
def compute_loss_score(model, tokenizer, text, device='cuda'):
    # Tokenize and prepare input_ids
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    # Prepare labels (next token prediction)
    labels = input_ids.clone()
    # Forward pass, get logits

    outputs = model(input_ids, labels=labels)
    # loss is already averaged over all tokens by default in HuggingFace
    # We want the **total** negative log-likelihood (sum over tokens)
    # outputs.loss is average, so multiply by number of tokens
    n_tokens = input_ids.shape[1]
    loss = -outputs.loss.item()
    return loss


@torch.no_grad()
def compute_min_k_score(model, tokenizer, text, device='cuda'):
    # Tokenize and prepare input_ids
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    # Prepare labels (next token prediction)
    labels = input_ids.clone()
    # Forward pass, get logits

    outputs = model(input_ids, labels=labels)
    logits = outputs.logits
    logits = logits.cpu()
    shift_logits = logits[..., :-1, :].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    all_probs = []
    for i in range(0, input_ids.size(1)-1):  # input_ids.size(1) is the sequence length
        # Extract log-probabilities for the ith token
        log_probabilities_for_token = log_probs[0, i, :]
        all_probs.append(log_probabilities_for_token)

    ngram_probs = []
    for i in range(0, len(all_probs) - 1 + 1, 1):
        ngram_prob = all_probs[i: i + 1]
        ngram_probs.append(np.mean(ngram_prob))
    min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * 0.2)]

    return -np.mean(min_k_probs)

def compute_zlib_loss_score(model, tokenizer, text, device='cuda'):
    # Tokenize and prepare input_ids
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    # Prepare labels (next token prediction)
    labels = input_ids.clone()
    # Forward pass, get logits

    outputs = model(input_ids, labels=labels)
    # loss is already averaged over all tokens by default in HuggingFace
    # We want the **total** negative log-likelihood (sum over tokens)
    # outputs.loss is average, so multiply by number of tokens
    n_tokens = input_ids.shape[1]
    loss = -outputs.loss.item()
    zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
    return loss / zlib_entropy


def test_auc(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # humaneval_data = load_dataset("HeyixInn0/Reorganized-humaneval",  split='train')
    # mbpp_data = load_dataset("HeyixInn0/Reorganized-mbpp", split='train')

    dataset_1 = load_from_disk("combined_dataset_1")
    dataset_2 = load_from_disk("combined_dataset_2")

    humaneval_loss_scores = []
    mbpp_loss_scores = []


    y_truth, y_pred, y_pred2, y_pred3 = [], [], [], []
    cnt = 0
    # Dataset that model is trained on
    for data in dataset_1:
        s = compute_loss_score(model, tokenizer, data['solution'], device)
        m = compute_min_k_score(model, tokenizer, data['solution'], device)
        z = compute_zlib_loss_score(model, tokenizer, data['solution'], device)
        score =  {
            "text": data['solution'],
            "score": s
        }
        humaneval_loss_scores.append(score)
        y_truth.append(1)
        y_pred.append(s)
        y_pred2.append(m)
        y_pred3.append(z)
        cnt += 1
        if cnt == 100:
            break
    cnt = 0
    for data in dataset_2:
        s = compute_loss_score(model, tokenizer, data['solution'], device)
        m = compute_min_k_score(model, tokenizer, data['solution'], device)
        z = compute_zlib_loss_score(model, tokenizer, data['solution'], device)
        score = {
            "text": data['solution'],
            "score": s
        }
        mbpp_loss_scores.append(score)

        y_truth.append(0)
        y_pred.append(s)
        y_pred2.append(m)
        y_pred3.append(z)
        cnt += 1
        if cnt == 100:
            break

    auc = roc_auc_score(y_truth, y_pred)
    min_k_auc = roc_auc_score(y_truth, y_pred2)
    z_auc = roc_auc_score(y_truth, y_pred3)

    print(f"{model_name} Loss Attack AUC: {auc}")
    print(f"{model_name} Min K Attack AUC: {min_k_auc}")
    print(f"{model_name} ZLib Attack AUC: {z_auc}")



if __name__ == "__main__":
    # Example usage
    model_name_list = [
        # "Qwen/Qwen2.5-0.5B",
        # "Qwen/Qwen2.5-Coder-0.5B",
        # "meta-llama/Llama-3.2-1B",
        # "deepseek-ai/deepseek-coder-1.3b-base",
        "./deepseek-coder-1.3b-base-finetuned-humaneval/1",
        # "./qwen2.5-0.5b-finetuned-humaneval/1",
        # "./qwen2.5-0.5b-finetuned-humaneval/2",
        # "./qwen2.5-0.5b-finetuned-humaneval/3",
        # "./qwen2.5-coder-0.5b-finetuned-humaneval/1"
        # "./qwen2.5-0.5b-finetuned-humaneval/1",
        # "./deepseek-coder-1.3b-base-finetuned-humaneval/3"
    ]
    for model_name in model_name_list:
        test_auc(model_name)
