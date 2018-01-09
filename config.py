class TrainingConfig(object):
    p = 0.9
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 50
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]

    fully_connected_layers = [1024, 1024]
    th = 1e-6


class Config(object):
    max_length = 200
    batch_size = 64
    no_of_classes = 1
    learning_rate = 0.5

    training = TrainingConfig()
    model = ModelConfig()


config = Config()
