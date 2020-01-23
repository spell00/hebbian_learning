from data_preparation.GeoParser import GeoParser
import pandas as pd


def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def print_infos():
    pass


def get_num_parameters():
    pass


def __main_gm__():
    meta_df = None
    unlabelled_meta_df = None
    
    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from files_destinations import home_path, destination_folder, data_folder, meta_destination_folder, \
        results_folder, plots_folder_path
    examples_list = ["mnist", "cifar10", "cifar100"]
    if example not in examples_list and not import_local_file and import_geo:
        g = GeoParser(home_path=home_path, geo_ids=geo_ids, unlabelled_geo_ids=unlabelled_geo_ids, bad_geo_ids=bad_geo_ids)
        g.get_geo(load_from_disk=load_from_disk)
        meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=True)
        unlabelled_meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=False)
        if translate is "y":
            for geo_id in geo_ids:
                g.translate_indices_df(geo_id, labelled=True)
            for geo_id in unlabelled_geo_ids:
                g.translate_indices_df(geo_id, labelled=False)
        meta_df, unlabelled_meta_df = adapt_datasets(meta_df, unlabelled_meta_df)

    is_example = False
    from list_parameters import z_dims, h_dims, num_elements, a_dims
    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows,a_dim=a_dim,
                                           num_elements=num_elements, use_conv=use_conv)

    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                        destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                        meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    if meta_df is not None:
        dgm.import_dataframe(meta_df, batch_size, labelled=True)
        if has_unlabelled:
            dgm.import_dataframe(unlabelled_meta_df, batch_size, labelled=False)
        else:
            dgm.import_dataframe(meta_df, batch_size, labelled=False)
    elif example in examples_list:
        dgm.load_example_dataset(dataset=example, batch_size=batch_size, labels_per_class=400)
        is_example = True
        print("PCA saved at: ", plots_folder_path)
        #meta_df = pd.DataFrame(dgm.train_ds)
        #ordination2d(meta_df, epoch="pre", dataset_name=dataset_name, ord_type="pca",
        #             images_folder_path=plots_folder_path, info=str(geo_ids)+str(unlabelled_geo_ids)+str(bad_geo_ids))

    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)
    dgm.set_data(labels_per_class=labels_per_class, is_example=is_example)
    dgm.cuda()

    if use_conv:
        dgm.set_conv_adgm_layers(is_hebb_layers=False, input_shape=img_shape)
    else:
        dgm.set_adgm_layers(h_dims=h_dims_classifier)

    # import the M1 in the M1+M2 model (Kingma et al, 2014)
    if load_vae:
        print("Importing the model: ", dgm.model_file_name)
        if use_conv:
            dgm.import_cvae()
        else:
            dgm.load_ae(load_history=False)
        #dgm.set_dgm_layers_pretrained()

    if resume:
        print("Resuming training")
        dgm.load_model()

    dgm.cuda()
    # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2 )

if __name__ == "__main__":
    from parameters import *
    from list_parameters import *

    __main_gm__()
