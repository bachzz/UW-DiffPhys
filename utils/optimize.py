import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
    

# def get_optimizer_uw(config, parameters_1, parameters_2):
#     if config.optim.optimizer == 'Adam':
#         return optim.Adam(list(parameters_1)+list(parameters_2), lr=config.optim.lr, weight_decay=config.optim.weight_decay,
#                           betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
#     elif config.optim.optimizer == 'RMSProp':
#         return optim.RMSprop(list(parameters_1)+list(parameters_2), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
#     elif config.optim.optimizer == 'SGD':
#         return optim.SGD(list(parameters_1)+list(parameters_2), lr=config.optim.lr, momentum=0.9)
#     else:
#         raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
