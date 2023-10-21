import torch
from attack_steps import L2Step, LinfStep


def forward(model, x, normalization, first_logit=True):
    assert normalization in ['cifar10', 'imagenet']
    if normalization == 'cifar10':
        # https://github.com/MadryLab/robustness/blob
        # /ca52df73bb94f5a3abb74d95b82a13589354a83e/robustness/datasets.py#L293
        mean = torch.as_tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype,
                               device=x.device)
        std = torch.as_tensor([0.2023, 0.1994, 0.2010], dtype=x.dtype,
                              device=x.device)
    else:
        # Use ImageNet normalization 
        # https://pytorch.org/docs/stable/torchvision/models.html
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=x.dtype,
                               device=x.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=x.dtype,
                              device=x.device)

    logits = model((x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1))
    return logits[:, 0] if first_logit else logits


def perturb(model, x, norm, eps, step_size, steps, normalization,
            random_start=False, descent=False):
    """Perform PGD attack."""
    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x

    x0 = x.clone().detach()
    step_class = L2Step if norm == 'L2' else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for i in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = forward(model, x, normalization)
        loss = logits.sum()
        if descent:
            loss = -logits.sum()
        # grad, = torch.autograd.grad(loss, [x])
        grad, = torch.autograd.grad(outputs=loss, inputs=[x], \
                                    grad_outputs=None, retain_graph=False, \
                                    create_graph=False, only_inputs=True, \
                                    allow_unused=False)
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x.clone().detach()
