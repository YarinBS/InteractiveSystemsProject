from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def get_embedding(text: str, tokenizer, model, device: torch.device, require_custom_mean_pooling=True):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)

    tokens = tokenizer(text, return_tensors="pt")
    tokens.to(device)

    with torch.no_grad():
        output = model(**tokens)

    if require_custom_mean_pooling:
        return mean_pooling(output, tokens['attention_mask'])
    else:
        mean_pooled = torch.mean(output['hidden_states'][-1], dim=1)
        return mean_pooled


def get_doc_embedding(docs, doc, tokenizer, model, device):
    doc_embeddings = {}
    for chunk in docs[doc]:
        emb = get_embedding(chunk, tokenizer, model, device, require_custom_mean_pooling=True)
        doc_embeddings[chunk] = emb
    return doc_embeddings


def get_top_texts(question: str,
                  docs: dict,
                  top_k: int,
                  similarity_threshold: float,
                  model,
                  tokenizer,
                  device: torch.device):
    print(f"Question: {question}")

    input_embeddings = get_embedding(text=question, tokenizer=tokenizer, model=model, device=device,
                                     require_custom_mean_pooling=True)

    docs_with_embeddings = {}
    for doc in docs.keys():
        docs_with_embeddings[doc] = get_doc_embedding(docs, doc, tokenizer, model, device)

    top_texts_and_scores = {f"None_{i+1}": -float('inf') for i in range(top_k)}

    for doc in docs_with_embeddings.keys():
        for i, (chunk_text, chunk_embedding) in enumerate(docs_with_embeddings[doc].items()):
            cos_sim = torch.nn.CosineSimilarity()
            if cos_sim(input_embeddings, chunk_embedding).item() > min(top_texts_and_scores.values()):
                # Remove current least-similar item
                least_similar = min(top_texts_and_scores, key=top_texts_and_scores.get)
                top_texts_and_scores.pop(least_similar)

                # Add the new one to all lists
                top_texts_and_scores[f"Doc {doc}___chunk {i+1}: {chunk_text}"] = cos_sim(input_embeddings, chunk_embedding).item()

    return [k for k in top_texts_and_scores.keys() if top_texts_and_scores[k] >= similarity_threshold]
