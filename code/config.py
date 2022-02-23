import os
import yaml
import shutil
import logging
from pathlib import Path


class ParseConfig(object):
    """
    Loads and returns the configuration specified in configuration.yml
    """
    def __init__(self):


        # 1. Load the configuration file ------------------------------------------------------------------------------
        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        except FileNotFoundError:
            logging.warning('Could not find configuration.yml')
            exit()




        # 2. Initializing ParseConfig object --------------------------------------------------------------------------
        self.model = conf_yml['model']
        self.dataset = conf_yml['dataset']
        self.active_learning = conf_yml['active_learning']
        self.train = conf_yml['train']
        self.metric = conf_yml['metric']
        self.experiment_settings = conf_yml['experiment_settings']
        self.architecture = conf_yml['architecture']
        self.viz = conf_yml['visualize']
        self.tensorboard = conf_yml['tensorboard']
        self.resume_training = conf_yml['resume_training']




        # 3. Extra initializations based on configuration chosen ------------------------------------------------------

        # Whether to load AuxNet or not
        if self.active_learning['algorithm'] in ['learning_loss', 'uncertainty_BN', 'uncertainty_KendallGal']:
            self.model['aux_net']['load'] = True
            assert self.active_learning['algorithm'] == self.model['aux_net']['method'], \
                "Same method should be chosen for active learning and aux net training."
        else:
            self.model['aux_net']['load'] = False


        # Number of convolutional channels for AuxNet
        if self.model['type'] == 'hourglass':
            if self.architecture['aux_net']['conv_or_avg_pooling'] == 'conv':
                self.architecture['aux_net']['channels'] = [self.architecture['hourglass']['channels']] * 5
                self.architecture['aux_net']['spatial_dim'] = [64, 32, 16, 8, 4]
            else:
                self.architecture['aux_net']['channels'] = [self.architecture['hourglass']['channels']]

        else:
            if self.architecture['aux_net']['conv_or_avg_pooling'] == 'conv':
                self.architecture['aux_net']['channels'] = self.architecture['hrnet']['STAGE3']['NUM_CHANNELS']
                self.architecture['aux_net']['spatial_dim'] = [
                    64 // (2**i) for i in range(self.architecture['hrnet']['STAGE3']['NUM_BRANCHES'])]
            else:
                self.architecture['aux_net']['channels'] = [self.architecture['hrnet']['STAGE4']['NUM_CHANNELS'][0]]


        # Number of heatmaps (or joints) based on the dataset
        if self.dataset['load'] == 'mpii':
            self.experiment_settings['num_hm'] = 16
            self.architecture['hrnet']['num_hm'] = 16
            self.architecture['hourglass']['num_hm'] = 16
            self.architecture['aux_net']['num_hm'] = 16

        else:
            assert self.dataset['load'] == 'lsp' or self.dataset['load'] == 'merged',\
                "num_hm defined only for 'mpii' and 'lsp' datasets"
            self.experiment_settings['num_hm'] = 14
            self.architecture['hrnet']['num_hm'] = 14
            self.architecture['hourglass']['num_hm'] = 14
            self.architecture['aux_net']['num_hm'] = 14



        # Number of output nodes for the aux_network
        if self.active_learning['algorithm'] == 'learning_loss':
            assert self.model['aux_net']['method'] == 'learning_loss', "AuxNet train method should be same as algorithm"
        if self.active_learning['algorithm'] == 'uncertainty_KendallGal':
            assert self.model['aux_net']['method'] == 'uncertainty_KendallGal', "AuxNet train method should be same as algorithm"

        if self.model['aux_net']['method'] == 'learning_loss':
            self.architecture['aux_net']['fc'].append(1)
        if self.model['aux_net']['method'] == 'uncertainty_KendallGal':
            self.architecture['aux_net']['fc'].append(self.architecture['aux_net']['num_hm'])


        # 1. Is aux_net inference required?  2. If train_auxnet_only, is auxnet set to train?
        self.use_auxnet = self.model['aux_net']['train'] or self.active_learning['algorithm'] in \
                          ['learning_loss', 'uncertainty_BN', 'uncertainty_KendallGal']
        if self.model['aux_net']['train_auxnet_only']: assert self.model['aux_net']['train'], \
            "Train aux_net only requires aux_net train = True"


        # Only HRNet, Hourglass supported
        assert self.model['type'] in ['hrnet', 'hourglass'], "Invalid Model type given: {}.".format(self.model['type'])

        # Resume training
        if self.resume_training:
            assert self.model['load'], "Resume training specified but model load == False"
            assert self.train, "Resume training requires train to be True."



        # if model load == False, then base method should be selected for sampling
        if not self.model['load']:
            assert self.active_learning['algorithm'] == 'base', "Sampler should be base since no model is loaded."

        if self.experiment_settings['all_joints']:
            assert self.experiment_settings['occlusion'], "Occlusion needs to be true if all joints is true."




        # 4. Create directory for model save path ----------------------------------------------------------------------
        self.experiment_name = conf_yml['experiment_name']
        i = 1
        model_save_path = os.path.join(self.model['save_path'], self.experiment_name + '_' + str(i))
        while os.path.exists(model_save_path):
            i += 1
            model_save_path = os.path.join(self.model['save_path'], self.experiment_name + '_' + str(i))

        logging.info('Saving the model at: ' + model_save_path)
        os.makedirs(os.path.join(model_save_path, 'model_checkpoints'))

        # Copy the configuration file into the model dump path
        code_directory = Path(os.path.abspath(__file__)).parent
        shutil.copytree(src=str(code_directory),
                        dst=os.path.join(model_save_path, code_directory.parts[-1]))

        self.model['save_path'] = model_save_path