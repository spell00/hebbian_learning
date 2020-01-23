import os
from utils.load_data import load_dataset
import torch
import torch.optim as optim


def run(train_loader, val_loader, test_loader, model_name, warmup, z1_size, z2_size, input_size, dataset_name, lr):
    if torch.cuda.is_available():
        cuda = True
        torch.cuda.manual_seed(1)

    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + model_name + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    print('create model')
    # importing model
    if model_name == 'vae':
        from models.generative.autoencoders.VAE import VAE
    elif model_name == 'vae_HF':
        from models.generative.autoencoders.VAE_HF import VAE
    elif model_name == 'vae_ccLinIAF':
        from models.generative.autoencoders.VAE_ccLinIAF import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(input_size, z1_size, number_of_flows=3)
    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ======================================================================================================================
    print('perform experiment')
    from utils.perform_experiment import experiment_vae
    experiment_vae(train_loader, val_loader, test_loader, model, optimizer, dir, model_name = model_name)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('vae_experiment_log.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    run()

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #