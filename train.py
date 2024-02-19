from argparse import ArgumentParser
import configparser
import wandb
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from dataloader import Dataloader_ours
from dataloader_nerfactor import Dataloader_nerfactor
from dataloader_dtu import Dataloader_dtu
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from model import Model

def main(config, wandb_logger):
    enable_val = config.getboolean('validation', 'enable_val')

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
    model.set_mode('train')

    if enable_val:
        print('Start training with validation')
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                patience=config.getint('training', 'early_stopping_patience'), verbose=True)

        trainer = Trainer(default_root_dir=log_dir,
                          max_epochs=config.getint('training', 'max_epoch'),
                          accelerator='gpu',
                          devices=-1,
                          precision=16,
                          logger=wandb_logger,
                          deterministic=True,  # reproducibility
                          callbacks=[early_stopping_callback],
                          fast_dev_run=False,  # debug, default False
                          overfit_batches=0,  # debug, default 0
                          profiler=None,  # debug, default None, option 'simple', 'advanced'
                          resume_from_checkpoint=None,
                          check_val_every_n_epoch=config.getint('validation', 'check_val_every_n_epoch'),
                          num_sanity_val_steps=3,  # default 3
                          limit_val_batches=3,  # default 1.0
                          )

    else:
        print('Start training without validation')
        dataloader.enable_val = False
        trainer = Trainer(default_root_dir=log_dir,
                          max_epochs=config.getint('training', 'max_epoch'),
                          accelerator='gpu',
                          devices=-1,
                          precision=16,
                          logger=wandb_logger,
                          deterministic=True,  # reproducibility
                          profiler=None,  # debug, default None, option 'simple', 'advanced'
                          limit_val_batches=0,  # disable validation
                          )


    trainer.fit(model, dataloader)
    trainer.save_checkpoint(Path(log_dir, 'checkpoint_no_correction.ckpt'))

    # # clear memory
    # del model, trainer
    # model = Model(config, wandb_logger)
    # model = model.load_from_checkpoint(Path(log_dir, 'checkpoint_no_correction.ckpt'))
    #
    # # calculate correction
    # dataloader.test_val_dataset = True
    # trainer = Trainer(default_root_dir=log_dir,
    #                   accelerator='gpu',
    #                   devices=-1,
    #                   precision=16,
    #                   logger=wandb_logger,
    #                   deterministic=True,  # reproducibility
    #                   limit_test_batches=3,  # default: 1.0
    #                   )
    # model.set_mode('cal_correction')
    # model.set_log_dir(log_dir)
    # trainer.test(model, dataloader)
    #
    # trainer.save_checkpoint(Path(log_dir, 'checkpoint.ckpt'))






if __name__ == '__main__':
    # add argparse
    parser = ArgumentParser()
    # default_config = './config/test_cube.ini'
    default_config = './config/test_lego.ini'
    parser.add_argument("--config", default=default_config, type=str)
    parser.add_argument("--max_epoch", default=0)
    args = parser.parse_args()
    # add configparse
    config = configparser.ConfigParser()
    config.read(args.config)

    # modify config
    if args.max_epoch != 0:
        config.set('training', 'max_epoch', args.max_epoch)

    # set up for weights and biases
    project = config.get('training', 'project')
    name = config.get('training', 'name')
    log_dir = f"./log/{project}/{name}/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)



    wandb.init(name=name, project=project, dir=log_dir)
    wandb_logger = WandbLogger(save_dir=log_dir)
    main(config, wandb_logger)

