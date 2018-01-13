class TrainingConfig(object):
    p = 0.9
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 1000
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[512, 5, 3],
                   [512, 5, 3],
                   [512, 2, None],
                   [512, 2, None],
                   [512, 2, None],
                   [512, 2, 3]]

    fully_connected_layers = [1024, 1024]
    th = 1e-6


class Config(object):
    max_length = 500
    batch_size = 128
    no_of_classes = 2
    learning_rate = 0.0001

    training = TrainingConfig()
    model = ModelConfig()


config = Config()
