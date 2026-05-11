import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataloader import get_input_dim, DATASET_INFO

def load_model(config):
    from models.v1_linear_ae import LinearAutoencoder
    from models.v2_convolutional_ae import ConvolutionalAutoencoder
    from models.v3_vae import VariationalAutoencoder

    dataset = config['dataset']
    info = DATASET_INFO[dataset]
    input_dim = get_input_dim(dataset)
    model_name = config['model']

    if model_name == 'v1_linear_ae':
        return LinearAutoencoder(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )

    elif model_name == 'v2_convolutional_ae':
        return ConvolutionalAutoencoder(
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )

    elif model_name == 'v3_vae':
        return VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_model_from_checkpoint(model_name, dataset, latent_dim, checkpoint_path):
    import torch

    config = {
        'model': model_name,
        'dataset': dataset,
        'latent_dim': latent_dim,
    }

    model = load_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model