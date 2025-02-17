from torch import nn

from hybrik.utils import Registry, build_from_cfg


SPPE = Registry('sppe')
print(SPPE.module_dict)
LOSS = Registry('loss')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        print('cfg is list')
        return nn.Sequential(*modules)
    else:
        print('cfg is not list')
        return build_from_cfg(cfg, registry, default_args)
        


def build_sppe(cfg):
    return build(cfg, SPPE)


def build_loss(cfg, **kwargs):
    default_args = dict()

    for key, value in kwargs.items():
        default_args[key] = value

    return build(cfg, LOSS, default_args=default_args)
