import torch
import torch.nn.functional as F

def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels):
    classes = torch.unique(support_labels)
    prototypes = torch.stack([support_embeddings[support_labels == c].mean(0) for c in classes])
    dists = torch.cdist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    acc = (log_p_y.argmax(dim=1) == query_labels).float().mean()
    return loss, acc
