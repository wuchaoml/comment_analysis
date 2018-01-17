class TrainingConfig(object):
    p = 0.5
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 10
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 5, 3],
                   [256, 5, 3],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, 3]]

    fully_connected_layers = [256, 128, 64]
    th = 1e-6


class Config(object):
    max_length = 128
    batch_size = 512
    no_of_classes = 2
    learning_rate = 0.00001
    beta = 7

    training = TrainingConfig()
    model = ModelConfig()


config = Config()

'''0.924
class TrainingConfig(object):
    p = 0.5
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 10
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[128, 5, 3],
                   [128, 5, 3],
                   [128, 2, None],
                   [128, 2, None],
                   [128, 2, None],
                   [128, 2, 3]]

    fully_connected_layers = [256, 128, 64]
    th = 1e-6


class Config(object):
    max_length = 128
    batch_size = 512
    no_of_classes = 2
    learning_rate = 0.00001
    beta = 3.5

    training = TrainingConfig()
    model = ModelConfig()


config = Config()
'''

'''0.9424 17k
class TrainingConfig(object):
    p = 0.5
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 10
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[128, 5, 3],
                   [128, 5, 3],
                   [128, 2, None],
                   [128, 2, None],
                   [128, 2, None],
                   [128, 2, 3]]

    fully_connected_layers = [128, 64]
    th = 1e-6


class Config(object):
    max_length = 128
    batch_size = 256
    no_of_classes = 2
    learning_rate = 0.00001
    beta = 0.01

    training = TrainingConfig()
    model = ModelConfig()


config = Config()

'''


'''0.948,step = 11k
class TrainingConfig(object):
    p = 0.5
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 10
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 5, 3],
                   [256, 5, 3],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, 3]]

    fully_connected_layers = [256, 128]
    th = 1e-6


class Config(object):
    max_length = 128
    batch_size = 256
    no_of_classes = 2
    learning_rate = 0.000025
    bata = 0.01

    training = TrainingConfig()
    model = ModelConfig()


config = Config()
'''


'''
class TrainingConfig(object):
    p = 0.5
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 10
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 5, 3],
                   [256, 5, 3],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, 3]]

    fully_connected_layers = [512, 256]
    th = 1e-6


class Config(object):
    max_length = 200
    batch_size = 256
    no_of_classes = 2
    learning_rate = 0.000025
    beta = 0.01

    training = TrainingConfig()
    model = ModelConfig()


config = Config()

'''
