from argparse import ArgumentParser
import configparser
import wandb
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from dataloader import Dataloader_ours
from dataloader_nerfactor import Dataloader_nerfactor
from dataloader_dtu import Dataloader_dtu
from pytorch_lightning import Trainer
from model import Model
import glob
import torch
import mitsuba as mi
import os
from metrics import metrics_test, metrics_relighting
mi.set_variant("cuda_ad_rgb")

def main(config, wandb_logger):


    dataset = config.get('training', 'dataset')
    assert dataset in ['ours', 'nerfactor', 'dtu']
    if dataset == 'ours':
        dataloader = Dataloader_ours(data_dir=config.get('training', 'data_dir'),
                                     data_name=config.get('training', 'data_name'))
    if dataset == 'nerfactor':
        dataloader = Dataloader_nerfactor(data_dir=config.get('training', 'data_dir'))
    if dataset == 'dtu':
        dataloader = Dataloader_dtu(data_dir=config.get('training', 'data_dir'))

    model = Model(config, wandb_logger)


    # load checkpoint
    checkpoint_path = Path(log_dir, 'checkpoint.ckpt')
    model = model.load_from_checkpoint(checkpoint_path)
    model.config = config  # update config

    if dataset == 'dtu':
        model.rgb_correction_linear = torch.ones_like(model.rgb_correction_linear)  # only for dtu


    assert model.rgb_correction_linear is not None
    model.rgb_correction_linear = model.rgb_correction_linear.cuda()
    model.set_log_dir(log_dir)

    trainer = Trainer(default_root_dir=log_dir,
                      accelerator='gpu',
                      devices=-1,
                      precision=16,
                      logger=wandb_logger,
                      deterministic=True,  # reproducibility
                      limit_test_batches=1.0,  # default: 1.0
                      )
    material_editing = config.getboolean('testing', 'material_editing')
    relighting = config.getboolean('testing', 'relighting')

    if config.getboolean('testing', 'test_val_dataset'):
        dataloader.test_val_dataset = True
        evaluate_dataset = 'val'
    else:
        dataloader.test_val_dataset = False
        evaluate_dataset = 'test'

    if material_editing:
        print('Start material editing')
        model.set_mode('material_editing')
        trainer.test(model, dataloader)
    else:
        if not relighting:
            print('Start testing')
            model.set_mode('test')
            trainer.test(model, dataloader)
            metrics_test(log_dir, evaluate_dataset)
        else:
            print('Start relighting')
            model.set_mode('relighting')
            env_dir = config.get('testing', 'env_dir')
            env_list = glob.glob(str(Path(env_dir, "*.hdr")))
            for env in env_list:
                env_name = os.path.basename(env)[:-4]
                print(f'Start relighting {env_name}')
                envmap_dict = {'type': 'envmap',
                               'filename': env,
                               'scale': 1,
                               'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                                           mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}
                emitter = mi.load_dict(envmap_dict)
                relighting_envmap = mi.traverse(emitter)['data'].torch()
                relighting_envmap[:,:,:3] *= model.rgb_correction_linear.permute(1,2,0)
                model.relighting_envmap = torch.nn.parameter.Parameter(relighting_envmap)
                model.relighting_env_name = env_name
                trainer.test(model, dataloader)
            metrics_relighting(log_dir, config.get('training', 'data_dir'), config.get('testing', 'env_dir'), evaluate_dataset)











if __name__ == '__main__':
    # add argparse
    parser = ArgumentParser()
    # default_config = './config/test_cube.ini'
    default_config = './config/ours_nerfactor/hotdog.ini'
    parser.add_argument("--config", default=default_config, type=str)
    parser.add_argument("--relighting", default=None)
    parser.add_argument("--test_val_dataset", default=True)
    args = parser.parse_args()
    # add configparse
    config = configparser.ConfigParser()
    config.read(args.config)

    # modify config
    if args.relighting is not None:
        config.set('testing', 'relighting', args.relighting)
    config.set('testing', 'test_val_dataset', args.test_val_dataset)



    # set up for weights and biases
    project = config.get('training', 'project')
    name = config.get('training', 'name')
    log_dir = f"./log/{project}/{name}/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if config.getboolean('testing', 'material_editing'):
        name = name + '_material_editing'
    else:
        if config.getboolean('testing', 'relighting'):
            name = name + '_relighting'
        else:
            name = name + '_test'
    wandb.init(name=name, project=project, dir=log_dir)
    wandb_logger = WandbLogger(save_dir=log_dir)
    # for each_section in config.sections():
    #     for (each_key, each_val) in config.items(each_section):
    #         wandb_logger.experiment.config.update({each_section + ':' + each_key: each_val})
    # seed_everything(42, workers=True)
    main(config, wandb_logger)
