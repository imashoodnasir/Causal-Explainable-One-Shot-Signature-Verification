def maml_update(model, task, loss_fn, alpha=0.01):
    support_data, query_data = task
    support_x, support_y = support_data
    query_x, query_y = query_data

    fast_weights = list(model.parameters())
    support_pred = model(support_x)
    loss = loss_fn(support_pred, support_y)
    grads = torch.autograd.grad(loss, fast_weights)
    
    fast_weights = [p - alpha * g for p, g in zip(fast_weights, grads)]

    query_pred = model(query_x, params=fast_weights)
    return loss_fn(query_pred, query_y)
