#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:37:22 2017

@author: simonpelletier


"""

# test for github
import os
import numpy as np
import GEOparse as geo
import pandas as pd
from utils.utils import overlap_mbk, remove_all_unique, remove_all_same, create_missing_folders, select_meta_number, rename_labels, get_user_int

def search_meta_file_name(filename="GSE33000_GSE22845_GSE12417",list_meta_files=["GSE12417_GSE22845","GSE33000"], ask=False):
    """
    :param filename:
    :param list_meta_files:
    :param ask:
    :return:

    filename="GSE33000_GSE22845_GSE12417"
    list_meta_files=["GSE12417","GSE22845","GSE33000","GSE22845_GSE12417","GSE12417_GSE22845_GSE33000"]
    ask=False
    result = search_meta_file_name(filename="GSE33000_GSE22845_GSE12417",list_meta_files=list_meta_files)
    result = search_meta_file_name(filename="GSE33000_GSE12417_GSE22845",list_meta_files=list_meta_files)

    print(result)

    """
    import numpy as np
    query_list = filename.split("_")
    filename_to_return = None
    for existing_file in list_meta_files:
        list_meta_file = existing_file.split("_")
        number_datasets_found = len(np.intersect1d(query_list, list_meta_file))
        if number_datasets_found == len(query_list):
            print("The file you were looking for is there")
            if ask:
                answer = input("Do you want to use it? [y/n]")
                if answer == "y" or answer == "yes":
                    use_file = True
                else:
                    use_file = False
            else:
                use_file = True

            if filename == existing_file and use_file:
                print("File found!")
                filename_to_return = filename
                break
            elif use_file:
                print("Defining the correct order of datasets in filename...")
                correct_order = [list_meta_file.index(x) for x in query_list]
                filename_to_return = "_".join(np.array(query_list)[correct_order])
                break
    return filename_to_return
class geoParser:
    def __init__(self,
                 home_path,
                 results_folder='results',
                 data_folder='data',
                 destination_folder='hebbian_learning_ann',
                 dataframes_folder="dataframes",
                 geo_ids = ["GSE12417","GSE22845"],
                 initial_lr=1e-3,
                 init="he_uniform", 
                 n_epochs=2 , 
                 batch_size=128,
                 hidden_size = 128, 
                 is_translate=True, 
                 silent=False
    ):
        """
        # initial_lr because it can change automatically as the epochs go with a SCHEDULER
        # for example : ReduceLROnPlateau reduces the learning rate (lr) when the results are not improved for
        # a number of iterations specified by the user.
        #
        Advantages:
            1- Can start learning very fast and then does fine tuning;
                -   Too high:
                        The accuracy will most likely reach its optimum faster, but might not be
                        as good as with a smaller LR

                -   Too small:
                        The accuracy will most likely reach a better optimum, but might not be quite long (if too low
                        might seem like its not learning. Too low and it might not learn anything at all)

        Pitfalls:
            1- LR reduction too frequent or too large               (same problem as LR too SMALL)
            2- LR reduction not fast enough or not large enough     (same problem as LR too HIGH)


        Examples:
        destination_folder = "/Users/simonpelletier/data/hebbian_learning_ann"
        initial_lr=1e-3
        init="he_uniform"
        n_epochs=2
        batch_size=128,
        hidden_size = 128
        is_translate=True
        silent=False

        """
        self.is_translate = is_translate
        self.silent = silent
        self.df = {}
        self.files_path = {}
        self.meta_file_path = None
        self.dataframes_folder = None

        # DATASETS IDS
        self.geo_ids = geo_ids

        # FOLDER NAMES
        
        self.data_folder = data_folder
        self.dataframes_folder = dataframes_folder
        self.destination_folder = destination_folder
        self.results_folder = results_folder

        # PATHS
        self.home_path = home_path
        self.data_folder_path = "/".join([self.home_path, self.destination_folder, self.data_folder]) + "/"
        self.results_folder_path = "/".join([self.home_path, self.destination_folder, self.results_folder]) + "/"
        self.dataframes_path = "/".join([self.data_folder_path, self.dataframes_folder])  + "/"
        self.translation_dict_path = "/".join([self.results_folder_path, "dictionaries"]) + "/"
        self.soft_path = self.data_folder_path + "/softs/"
        create_missing_folders(self.translation_dict_path)
        self.translation_results = {}
        # Hyperparameters
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.init = init
        self.hidden_size = hidden_size


    def getGEO(self,
               saveToDisk=True,
               is_load_from_disk=True
               ):

        """

        Function that get informations on Gene Expression Omnibus (GEO)
        :param geo_ids:
        :param merge_data:
        :param saveToDisk:
        :param is_load_from_disk:
        :param online:
        :return:

        from debug.get_parameters import *
        is_load_from_disk=True; saveToDisk=True
        g = geoParser(destination_folder=data_destination)
        g.getGEO(geo_ids,is_load_from_disk)
        dataframes_folder = "dataframes"
        self = g
        self.getGEO()
        """
        if len(self.geo_ids) == 0:
            print('WARNING! You must give at least one geo id! The object created will be empty')

        # The key is the geo_id and values the matrices
        self.df = {}
        flag = True
        for geo_id in self.geo_ids:
            print('Running:',geo_id)

            if is_load_from_disk:
                flag = self.loadGEO(geo_id)
                if flag:
                    print("The dataset will have to be built")
            if flag:
                print('Building the dataset ...')
                self.df[geo_id] = self.build_dataframe(geo_id,saveToDisk=saveToDisk)
                if self.df[geo_id] is not None:
                    print("The file containing the data for ",geo_id," was loaded successfully!")

    def loadGEO(self, geo_id='GSE22845', dataframes_folder="dataframes"):
        """

        :param geo_id:
        :return:

        Example:
        from debug.get_parameters import *

        dataframes_folder = "/Users/simonpelletier/data/hebbian_learning_ann"
        g = geoParser(destination_folder=data_destination)
        g.getGEO(geo_ids,is_load_from_disk)

        """
        import pandas as pd
        import numpy as np
        flag = False
        print('Loading ' + geo_id + ' ...')

        self.df_file_name = geo_id + '_dataframe.pickle.npy'
        create_missing_folders(self.dataframes_path)
        current_directory_list = os.listdir(self.dataframes_path)
        if self.df_file_name in current_directory_list:
            print("File found at location:",self.data_folder_path + "/" + self.df_file_name)
            self.df[geo_id] = pd.read_pickle(self.dataframes_path+"/"+self.df_file_name)

            if sum(sum(np.isnan(self.df[geo_id].values)).tolist()) > 0:
                print("Nans found. They are all replaced by 0")
                self.df[geo_id][np.isnan(self.df[geo_id] )] = 0

        else:
            print(self.df_file_name ,' NOT FOUND in ', self.dataframes_path)
            print(current_directory_list)
            flag = True
        return flag
    def make_metadata_matrix(self,gse,merged_values):
        """
        function to build a 2d matrix of the metadata (used to see which not to display)
        :return:
        example
        import GEOparse as geo
        import numpy as np
        from geoParser import geoParser
        from utils import remove_all_unique, remove_all_same, create_missing_folders

        gse = geo.GEOparse.get_GEO(geo="GSE12417",destdir="/home/simon/data/hebbian_learning_ann/",silent=True)
        g = geoParser("GSE12417")

        geo_id='GSE33000'
        meta_dict = g.make_metadata_matrix(gse)
        meta_dict = remove_all_unique(meta_dict)
        meta_dict = remove_all_same(meta_dict)
        print(meta_dict)
        """

        allInfos = np.empty(shape=(len(list(gse.gsms[list(gse.gsms.keys())[0]].metadata.values())),len(merged_values.columns))).tolist()
        meta_names = []

        for l,lab in enumerate(list(merged_values.columns)):
            infos_label = list(gse.gsms[lab].metadata.values())
            for i,info in enumerate(infos_label):
                try:
                    allInfos[i][l] = ' '.join(info)
                except:
                    allInfos[i][l] = info[0]
                meta_names += [list(gse.gsms[lab].metadata.keys())[i]]
        meta_dict = dict(zip(meta_names,allInfos))
        return meta_dict

    def rename_according_to_metadata(self,gse,meta_dict):
        """
        Renames the groups according to informations in the samples metadata
        :param gse: object
        :return: A list of the new samples names

        import GEOparse as geo
        import numpy as np
        from utils import select_meta_number
        from geoParser import geoParser

        home_path = "/Users/simonpelletier/"
        results_folder = "results"
        data_folder = "data"
        destination_folder = "hebbian_learning_ann"
        geo_id = "GSE33000"
        geo_ids = ["GSE33000"]

        g = geoParser(data_folder_path)

        gse = geo.GEOparse.get_GEO(geo=geo_id,destdir=g.destination_folder,silent=g.silent)
        merged_values = gse.merge_and_average(gse.gpls[next(iter(gse.gpls))], "VALUE", "GB_ACC", gpl_on="ID", gsm_on="ID_REF")


        g.merge_len = merged_values.shape[1]
        g.df[geo_id] = merged_values


        meta_dict = g.make_metadata_matrix(gse,merged_values)
        labels, meta_dict = g.rename_according_to_metadata(gse, meta_dict)




        """

        accept_names = False

        while not accept_names:
            meta_dict = select_meta_number(meta_dict)
            meta_dict = remove_all_unique(meta_dict)
            meta_dict = remove_all_same(meta_dict)

            number = get_user_int(meta_dict)
            meta_samples = meta_dict[list(meta_dict.keys())[number]]

            try:
                nsamples = int(input("How many samples names you want to see?"))
            except:
                print("Not a valid number, displaying 1...")
                nsamples = 1
            for i in range(nsamples):
                print(meta_samples[i])
            satisfied = input('Are you satisfied with your seletection? (y/n)')
            if len(meta_samples[0]) > 1 and type(meta_samples[0]) != str:
                print("There is a list inside the list! Which one of the following options do you want (the options "
                      "represent the label of one of the samples)?")
                number2 = get_user_int(meta_samples[0])
                meta_samples = []
                for i,sample in enumerate(gse.gsms):
                    elements = list(list(gse.gsms.values())[i].metadata.values())
                    meta_samples.append(elements[number2])

            if satisfied == 'y':
                accept_names = True
        return meta_samples, meta_dict

    def build_dataframe(self,geo_id='GSE22845',saveToDisk=True):

        """
        The labels are found in the metadata of merged object

        ** Not sure if still true, old comment**
            There is a weird behavior here. The metadata values are never in the same number,
            the metadata containing the disease state is also the longest, so the longest is found first
        ** Not sure if still true, old comment**

        :param geo_id: ID found on NCBI's database
            EXAMPLE: GSE12417 -> found here -> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12417
        :param saveToDisk (optional):

        EXAMPLE
        g = get_example_datasets(geo_ids = ["GSE12417","GSE22845"], home_path="/Users/simonpelletier/", is_load_from_disk=True)
        g.getGEO(geo_ids, is_load_from_disk=is_load_from_disk)

        """
        create_missing_folders(self.soft_path)
        gse = geo.GEOparse.get_GEO(geo=geo_id,destdir=self.soft_path,silent=self.silent)
        gsm_on_choices = list(gse.gsms[list(gse.gsms.keys())[0]].columns.index)
        gpl_on_choices = list(gse.gpls[list(gse.gpls.keys())[0]].columns.index)

        print( str(len(gsm_on_choices)) +" Choices are available for GSM")
        gsm_on_selection = get_user_int(gsm_on_choices)
        gsm_on = gsm_on_choices[gsm_on_selection]
        print(str(len(gpl_on_choices)) +" Choices are available for GPL. You must select: ")
        print("1 - An annotation for GPL")
        print("2 - (optional) The annotation you want the row names to take")

        gpl_on_selection = get_user_int(gpl_on_choices)
        gpl_on = gpl_on_choices[gpl_on_selection]

        val_selection = get_user_int(gpl_on_choices)
        val = gpl_on_choices[val_selection]

        merged_values = gse.merge_and_average(gse.gpls[next(iter(gse.gpls))], "VALUE", val,
                                gpl_on_choices, gpl_on=gpl_on, gsm_on=gsm_on)
        merged_values.values[np.isnan(merged_values.values)] = 0

        self.merge_len = merged_values.shape[1]
        self.df[geo_id] = merged_values


        meta_dict = self.make_metadata_matrix(gse,merged_values)
        labels, meta_dict = self.rename_according_to_metadata(gse, meta_dict)

        labels = ["".join(label) for label in labels]

        labels = rename_labels(labels)

        self.df[geo_id].columns = labels

        if len(labels) > merged_values.shape[1]:
            prompt = input("Duplicates were detected. Do you want to keep only the first labels (only say yes if you are sure, "
                  "or the results could be wrong) [y/n]")
            print("Labels",len(labels))
            print("Labels",merged_values.shape)

            if prompt =="y":
                labels = labels[:merged_values.shape[1]]
            else:
                exit()
        if saveToDisk:
            create_missing_folders(path=self.dataframes_path)
            self.files_path[geo_id] = self.dataframes_path + '/' + geo_id + '_dataframe.pickle.npy'
            print("Saving to " + self.files_path[geo_id])

            self.df[geo_id].to_pickle(self.files_path[geo_id])

        return merged_values

    def save(self, filename, merged_values):
        merged_values.to_pickle(filename)

    def remove(self, geo_id='GSE33000'):
        """

        USELESS??
        Removes a dataset
        :param geo_id:
        :return:
        """
        if geo_id not in self.geo_ids:
            print('The object does not contain the GEO ID specified')
        else:
            if len(self.geo_ids) <= 1:
                print('WARNING! The last dataset is being removed (object will be empty) ')

    def merge_datasets(self, fill_missing=True, is_load_from_disk=True, meta_destination_folder="meta_pandas_dataframes"):
        """

        :param fill_missing: if True, the missing rows will be replaced with 0s for all samples of that dataset.
                The algorythm might be able to do good even without some information. Otherwise, the list might get very small

        from utils import get_example_datasets, create_missing_folders
        fill_missing=True
        geo_ids = ["GSE12417","GSE22845"]
        g = get_example_datasets(geo_ids, home_path="/home/simon/", is_load_from_disk=True)
        g.getGEO(geo_ids, is_load_from_disk=True)
        g.merge_datasets(fill_missing=True)

        """

        print("Preparing for merging the selected datasets...")
        import os
        import pandas as pd
        import numpy as np
        self.meta_destination_folder = meta_destination_folder
        self.meta_data_folder_path = "/".join([self.data_folder_path, meta_destination_folder])
        create_missing_folders(self.meta_data_folder_path)
        meta_filename = "_".join(self.geo_ids)

        count = 0
        n_samples_list = [len(self.df[geo_id].columns) for geo_id in self.geo_ids ]
        total_iteration = -n_samples_list[0]
        meta_file = search_meta_file_name(meta_filename,list_meta_files = os.listdir(self.meta_data_folder_path))

        if meta_file is None or is_load_from_disk is False:
            for i in range(len(n_samples_list)):
                for j in range(i,len(n_samples_list)):
                    total_iteration += n_samples_list[j]
            for g,geo_id in enumerate(self.geo_ids):
                if g == 0:
                    meta_df = self.df[geo_id]
                else:
                    if fill_missing:
                        # FIRST find which rows are not already in the meta_df
                        union_rows = np.union1d(self.df[geo_id].index, meta_df.index)
                        all_cols = np.concatenate((self.df[geo_id].columns, meta_df.columns))
                        new_df = pd.DataFrame(0.00000, index=union_rows, columns=all_cols)
                        in1d_rows = np.in1d(new_df.index,self.df[geo_id].index)
                        in1d_rows_meta = np.in1d(new_df.index, meta_df.index)
                        ind_df_rows = np.array(list(range(len(in1d_rows))))[in1d_rows]
                        ind_meta_rows = np.array(list(range(len(in1d_rows_meta))))[in1d_rows_meta]

                        # SPEED BOTTLENECK
                        # TODO IS this optimal?
                        for c,col in enumerate(self.df[geo_id].columns):
                            count += 1
                            if count % 10 == 0:
                                print("Merge progress {:2.0%}".format(count/total_iteration), end="\r")
                            for r,row in enumerate(ind_df_rows):
                                new_df.values[row, c] = self.df[geo_id].values[r,c]

                        for c,col in enumerate(meta_df.columns):
                            count += 1
                            if count % 10 == 0:
                                print("Merge progress {:2.0%}".format(count/total_iteration), end="\r")
                            for r,row in enumerate(ind_meta_rows):
                                new_df.values[row, c] = meta_df.values[r,c]

                    else:
                        df_intersection_meta = np.in1d(self.df[geo_id].index, meta_df.index)
                        meta_intersection_df = np.in1d(meta_df.index, self.df[geo_id].index)
                        tmp = self.df[geo_id][df_intersection_meta]
                        meta_df = meta_df[meta_intersection_df]
                        new_df = pd.concat([tmp, meta_df], axis=1)

                try:
                    assert len(self.meta_df.index) == len(set(self.meta_df.index))
                except:
                    print("CONTAINS DUPLICATED ROWNAMES")
            self.meta_filename = meta_filename
            self.meta_file_path = '/'.join([self.meta_data_folder_path, self.meta_filename])
            self.meta_df = new_df
            self.meta_df.to_pickle(self.meta_file_path)
        else:
            self.meta_filename = meta_file
            self.meta_file_path = '/'.join([self.meta_data_folder_path, self.meta_filename])
            self.meta_df = pd.read_pickle(self.meta_file_path)
            print("Merged dataset imported from disk.")
    def translate(self, geo_id, old_ids='refseq_mrna', new_ids='uniprot_gn', load=False):
        import subprocess
        translation_destination = "/".join([self.data_folder_path, "translation_results"]) + "/"
        self.translation_destination = translation_destination
        dictionary_path = self.dictionary_path = "/".join([self.data_folder_path, "dictionaries"])
        create_missing_folders(translation_destination)
        create_missing_folders(dictionary_path)
        filename = geo_id + "_" + old_ids + '.txt'
        try:
            os.remove(translation_destination + "/" + filename)
        except OSError:
            pass
        os.mknod(translation_destination + "/" + filename)
        f = open(translation_destination + "/" + filename, "w")
        for id in list(self.meta_df.index):
            f.write(id+"\n")

        f.close()

        translation_filename = old_ids + "2" + new_ids + "_" + geo_id + ".txt.npy"
        self.translation_results[geo_id] = translation_destination
        create_missing_folders(self.translation_results[geo_id])
        if translation_filename not in translation_destination or load is False:
            print("Translating", geo_id, "from", old_ids, "to", new_ids, "...")
            call = ["./biomart_api.R", translation_destination, geo_id, old_ids, new_ids]
            subprocess.call(call)
        output_file = translation_destination + "/" + geo_id + "_" +old_ids+"2"+new_ids+".txt"
        file = open(output_file, "r")
        names_translated = file.readlines()

        print(names_translated)
        #print("LOADING...")
        #self.df[geo_id].index = np.load(self.translation_destination + translation_filename)


