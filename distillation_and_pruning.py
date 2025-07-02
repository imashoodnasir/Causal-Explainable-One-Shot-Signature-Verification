def distillation_loss(student_logits, teacher_logits, T=3):
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * (T * T)

def prune_weights(model, threshold=0.05):
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = (param.abs() > threshold).float()
            param.data *= mask
