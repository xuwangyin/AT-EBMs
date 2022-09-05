from pathlib import Path
import os
import torch


def set_bn_eval(module):
    for submodule in module.modules():
        if 'batchnorm' in submodule.__class__.__name__.lower():
            submodule.train(False)


def set_train(model):
    """Disable batch normalization when training."""
    model.train()
    set_bn_eval(model)


def set_eval(model):
    model.eval()


def save_model(model, savename):
    Path(os.path.dirname(savename)).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), savename)
    print(f'saved {savename}')


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    # from https://github.com/clovaai/stargan-v2/blob/875b70a150609e8a678ed8482562e7074cdce7e5/core/solver.py#L282
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.reshape(batch_size, -1).sum(1).mean(0)
    return reg
