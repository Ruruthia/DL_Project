from dl.models.resnet import ResNet18Lit
from dl.models.vanilla_cnn import CNNLit


def create_model(model_type, config):
    if model_type == "resnet":
        model = ResNet18Lit(config)
    elif model_type == "cnn":
        model = CNNLit(config)
    else:
        raise NotImplementedError("Possible model types are resnet or cnn!")
    return model