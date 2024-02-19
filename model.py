from tools.tcnn_tools import get_tcnn_config
from pytorch_lightning import LightningModule
import tinycudann as tcnn
import torch
import torchvision
import mitsuba as mi
from tools.mitsuba_tools import generate_camera_rays_randomly, generate_camera_rays_chunk, create_mitsuba_scene_envmap, \
    create_mitsuba_sensor, PowerHeuristic, m2t
from tools.color_mapping_blender import linear2srgb, srgb2linear
from BRDF import LambertianReflection, MicrofacetReflection, GlossyBRDF
import drjit as dr
import time
from pathlib import Path

mi.set_variant("cuda_ad_rgb")


class Model(LightningModule):
    def __init__(self, config, wandb_logger):
        super().__init__()
        # init parameters
        self.config = config
        self.wandb_logger = wandb_logger

        # init mode
        self.mode = 'train'
        self.log_dir = None
        self.relighting_env_name = None

        # read config
        mesh_path = config.get('training', 'mesh_path')
        envmap_path = config.get('training', 'gt_env_path')
        inten = config.getint('training', 'gt_env_inten')
        self.spp_training = config.getint('training', 'spp')
        self.spp_validation = config.getint('validation', 'spp')
        self.spp_test = config.getint('testing', 'spp')
        self.max_bounce = config.getint('training', 'max_bounce')
        self.n_pixel_train = config.getint('training', 'n_pixel')
        self.n_pixel_val = config.getint('validation', 'n_pixel')
        self.n_pixel_test = config.getint('testing', 'n_pixel')
        self.specular_type = config.get('training', 'specular_type')

        # init specular type
        assert self.specular_type in ['GGX', 'Phong', 'None']
        if self.specular_type == 'GGX':
            print("Using GGX specular BRDF")
        elif self.specular_type == 'Phong':
            print("Using Phong specular BRDF")
        else:
            print("Using no specular BRDF")


        # create mitsuba scene
        self.scene = create_mitsuba_scene_envmap(mesh_path, envmap_path, inten)

        # create tcnn mlp network, output = [R,G,B]
        tcnn_config = get_tcnn_config(mlp_depth=config.getint('mlp', 'mlp_depth'),
                                      mlp_width=config.getint('mlp', 'mlp_width'),
                                      output_activation="Sigmoid")
        self.mlp = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=5,
                                                 encoding_config=tcnn_config["encoding"],
                                                 network_config=tcnn_config["network"])

        # loss
        self.mse = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()

        # load some methods
        self.ToPILImage = torchvision.transforms.ToPILImage()

        # ini variables for dr ad_wrap
        [self.emitter, self.si, self.ds, self.active_b, self.active_em, self.sampler] = [None, None, None, None, None,
                                                                                         None]
        # create envmap
        params = mi.traverse(self.scene)
        key = 'emitter.data'
        env_map_gt = params[key].torch()
        self.env_map_gt = env_map_gt.detach().clone()
        self.train_env = config.getboolean('training', 'train_env')
        self.init_env_by_gt = config.getboolean('training', 'init_env_by_gt')
        if not self.train_env:
            # env map is fixed and equal to gt
            self.env_map = env_map_gt.detach().clone()
        else:
            if self.init_env_by_gt:
                # env map is initialized by gt
                env_map = env_map_gt.detach().clone()
            else:
                # env map is initialized by a dark ambient light
                env_map = torch.ones_like(env_map_gt)
                env_map[:, :, :3] *= 0.1
            # update env_map for mitsuba emitter
            self.env_map = torch.nn.parameter.Parameter(env_map)  # add env_map to self.parameters()
            params[key] = self.env_map * 1
            params.update()

        # register RGB correction
        self.rgb_correction_linear = None

        # save hyperparameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def set_mode(self, mode):
        assert mode in ['train', 'val', 'test', 'cal_correction', 'relighting', 'material_editing']
        self.mode = mode

    def set_log_dir(self, log_dir):
        self.log_dir = Path(log_dir, 'test')
        self.log_dir.mkdir(parents=True, exist_ok=True)


    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_envmap(n): return n == 'env_map'

        lr = self.config.getfloat('training', 'learning_rate')
        factor = self.config.getfloat('training', 'learning_rate_factor_envmap')
        grouped_parameters = [
            {"params": [p for n, p in params if is_envmap(n)], 'lr': lr * factor},
            {"params": [p for n, p in params if not is_envmap(n)], 'lr': lr},
        ]

        return torch.optim.Adam(grouped_parameters, lr=lr)

    ############################ Rendering ############################
    @dr.wrap_ad(source='torch', target='drjit')
    def emitter_eval(self, env_map):
        # pass env_map gradients through mitsuba methods
        params = mi.traverse(self.emitter)
        _key = 'data'
        params[_key] = env_map
        params.update()
        active = self.active_b & self.ds.emitter.is_environment()
        emitter_val = self.emitter.eval(self.si, active)
        result = dr.zeros(mi.TensorXf, shape=dr.shape(emitter_val))
        result[0] = emitter_val.x
        result[1] = emitter_val.y
        result[2] = emitter_val.z
        return result

    def forward(self, scene, sampler, ray, active=mi.Bool(True)):
        N_rays = dr.shape(ray.o)[1]

        # record ini
        record = {}
        record['Rd'] = torch.zeros((N_rays, 3), dtype=torch.float32, device='cuda')
        record['Rs'] = torch.zeros((N_rays, 1), dtype=torch.float32, device='cuda')
        record['roughness_x'] = torch.zeros((N_rays, 1), dtype=torch.float32, device='cuda')

        emitter = scene.emitters()[0]  # we have only one emitter
        emitter_sampling = True
        m_hide_emitters = True
        m_max_depth = self.max_bounce + 1  # 1 for seeing light source only, 2 for 1 bounce

        result = torch.zeros((N_rays, 3), dtype=torch.float32, device='cuda')
        throughput = torch.tensor([1], dtype=torch.float32, device='cuda')
        depth = 0
        valid_ray = mi.Bool(False)  # only rays that hit the object will be counted as valid

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = torch.tensor([1], dtype=torch.float32,
                                     device='cuda')  # bsdf_sample.pdf; pdf of sampling direction sampled from bsdf

        while True:

            si = scene.ray_intersect(ray, active)

            if depth == 0:
                valid_ray = si.is_valid()

            # ---------------------- Direct emission and BSDF Sampling Evaluation----------------------
            if dr.any(dr.neq(si.emitter(scene, active), None)):  # if any ray hit an emitter
                ds = mi.DirectionSample3f(scene, si, prev_si)
                em_pdf = 0.

                if depth != 0:
                    if emitter_sampling:
                        # Determine probability of having sampled that same direction using Emitter sampling.
                        em_pdf = scene.pdf_emitter_direction(prev_si, ds, active)
                        em_pdf = m2t(em_pdf)

                # execute if m_hide_emitters == False for the 0 bounce or depth > 0
                if not (depth == 0 and m_hide_emitters):
                    active_b = active & (mi.Float(prev_bsdf_pdf.flatten().float()) > 0.)

                    # Compute MIS weight for emitter sample from previous bounce
                    mis_bsdf = PowerHeuristic(prev_bsdf_pdf, em_pdf)  # 1 if depth==0
                    mis_bsdf = torch.where(m2t(active_b).bool(), mis_bsdf, torch.zeros_like(mis_bsdf))

                    if (not self.train_env) or (not self.mode == 'train'):
                        # if not training env, or not training the model, use mitsuba to eval emitter
                        emitter_val = ds.emitter.eval(si, active_b)
                        emitter_val = m2t(emitter_val, unsqueeze=False)
                        emitter_val = torch.clip(emitter_val, 0)
                    else:
                        env_map = self.env_map * 1
                        [self.ds, self.si, self.active_b, self.emitter] = ds, si, active_b, emitter
                        emitter_val = self.emitter_eval(env_map)

                        emitter_val = emitter_val.swapaxes(0, 1)
                        emitter_val = torch.clip(emitter_val, 0)

                    result = throughput * emitter_val * mis_bsdf + result  # beta * Li * w

            # Continue tracing the path at this point?
            active_next = (depth + 1 < m_max_depth) & si.is_valid()
            if not dr.any(active_next):  # all rays are inactive
                break

            # configurate BRDF
            xsurf = si.p.torch()
            mlp_output = self.mlp(xsurf)
            Rd = mlp_output[:, 0:3]
            Rs = mlp_output[:, 3].unsqueeze(1)
            roughness_x = mlp_output[:, 4].unsqueeze(1) * 0.98 + 0.02  # roughness_x in [0.02, 1]
            if self.mode == 'material_editing':
                Rs = torch.ones_like(Rs) * self.config.getfloat('testing', 'Rs')
                roughness_x = torch.ones_like(roughness_x) * self.config.getfloat('testing', 'roughness')
                Rd = torch.ones_like(Rd) * 0.3
            if self.specular_type == 'GGX':
                brdf = GlossyBRDF(Rd, Rs, roughness_x, specular_type='GGX')
            elif self.specular_type == 'Phong':
                brdf = GlossyBRDF(Rd, Rs, roughness_x, specular_type='Phong')
            else:
                brdf = LambertianReflection(Rd)
                Rs = torch.zeros_like(Rs)
                roughness_x = torch.zeros_like(roughness_x)



            # record
            if depth == 0:
                record['Rd'] = Rd.clone().detach()
                record['Rs'] = Rs.clone().detach()
                record['roughness_x'] = roughness_x.clone().detach()

                # smoothness:
                if self.config.getboolean('training', 'use_smoothness') and self.mode == 'train':
                    noise = torch.normal(mean=xsurf, std=0.01)
                    mlp_output_noise = self.mlp(noise)
                    record['Rd_noise'] = mlp_output_noise[:, 0:3]
                    record['Rs_noise'] = mlp_output_noise[:, 3].unsqueeze(1)
                    record['roughness_x_noise'] = (mlp_output_noise[:, 4].unsqueeze(1) * 0.98 + 0.02)

            if emitter_sampling:
                # ---------------------- Emitter sampling ----------------------
                active_em = active_next

                if dr.any(active_em):  # a not necessary step in our mirror-free setup

                    ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(active_em), True,
                                                                   active_em)
                    em_weight = m2t(em_weight, unsqueeze=False)
                    em_weight = torch.clip(em_weight, 0)

                    active_em = active_em & dr.neq(ds.pdf, 0.)  # mask out unvalid wo (ds.d)

                    # Query the BSDF for that emitter-sampled direction
                    wo = si.to_local(ds.d)

                    # Determine BSDF value and probability of having sampled that same direction using BSDF sampling.
                    bsdf_val, bsdf_pdf = brdf.eval_pdf(si.wi, wo, active_em)

                    mis = PowerHeuristic(m2t(ds.pdf), bsdf_pdf)
                    mis = torch.where(m2t(active_em).bool(), mis, torch.zeros_like(mis))
                    result = throughput * bsdf_val * em_weight * mis + result

            # ----------------------- BSDF sampling -----------------------
            wo, bsdf_pdf, bsdf_weight = brdf.sample(si.wi, sampler.next_2d(active), sampler.next_1d(active), active)

            ray = si.spawn_ray(si.to_world(wo))

            # ------ Update loop variables based on current interaction ------
            throughput = throughput * bsdf_weight  # bsdf_weight = f(wo, wi) * cos_theta / scatteringPdf
            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_pdf

            # -------------------- Stopping criterion ---------------------
            depth += 1
            active = active_next

        return result, valid_ray, record

    ############################ Training ############################
    def training_step(self, batch, batch_index):
        self.set_mode('train')
        spp = self.spp_training
        n_pixel = self.n_pixel_train

        # load data
        color_gt = batch['image_gt'][0]  # [3, imh, imw]
        color_gt = color_gt.reshape(3, -1)  # [3, imh, imw]
        metadata = batch['metadata']

        # create mitsuba sensor and generate camera rays
        sensor = create_mitsuba_sensor(metadata, spp)
        ray, sampler, index = generate_camera_rays_randomly(sensor, spp, n_pixel=n_pixel)


        # render
        color_predict, valid_ray, record = self(self.scene, sampler, ray)  # [n_pixel*spp, 3]
        # compute rgb for each pixel
        valid_ray = valid_ray.torch().bool()
        # if valid_ray.shape[0] == 1:
        #     valid_ray = valid_ray.expand(color_predict.shape[0])  # in case all rays are unvalid
        color_predict[~valid_ray, :] = 1
        color_predict = torch.mean(color_predict.reshape(n_pixel, spp, 3), dim=1)  # [n_pixel, 3], average over spp
        color_predict = linear2srgb(color_predict).swapaxes(0, 1)  # [3, n_pixel]

        # compute color loss
        color_loss = self.mse(color_predict, color_gt[:, index])  # in sRGB space
        self.log('color_loss', color_loss, on_epoch=True)

        # compute smoothness loss
        if self.config.getboolean('training', 'use_smoothness'):
            smoothness_loss = 0
            outputs = [record['Rd'], record['Rs'], record['roughness_x']]
            noises = [record['Rd_noise'], record['Rs_noise'], record['roughness_x_noise']]
            smoothness_weights = [self.config.getfloat('training', 'smoothness_Rd'),
                                  self.config.getfloat('training', 'smoothness_Rs'),
                                  self.config.getfloat('training', 'smoothness_roughness')]
            for i, j, k in zip(outputs, noises, smoothness_weights):
                # relative_loss = self.relative_loss(i, j, valid_ray)
                # smoothness_loss = (smoothness_loss + relative_loss * k)
                smoothness_loss = (smoothness_loss + self.mse(i[valid_ray, :], j[valid_ray, :]) * k)
            self.log('smoothness_loss', smoothness_loss, on_epoch=True)

            # env map smoothness
            if self.config.getfloat('training', 'smoothness_env') > 0:
                dx = torch.roll(self.env_map, shifts=1, dims=1) - self.env_map  # HxWx3
                dy = torch.roll(self.env_map, shifts=1, dims=0) - self.env_map  # HxWx3
                env_smoothness_loss = self.mse(dx, torch.zeros_like(dx)) + self.mse(dy, torch.zeros_like(dy))
                env_smoothness_loss = env_smoothness_loss * self.config.getfloat('training', 'smoothness_env')
                self.log('env_smoothness_loss', env_smoothness_loss, on_epoch=True)

            total_loss = color_loss + smoothness_loss + env_smoothness_loss
            self.log('total_loss', total_loss, on_epoch=True)
            return total_loss
        else:
            return color_loss


    # def relative_loss(self, i, j, valid_ray):
    #     relative_loss = (i[valid_ray, :] - j[valid_ray, :]) / torch.maximum(i[valid_ray, :], j[valid_ray, :])
    #     relative_loss[torch.maximum(i[valid_ray, :], j[valid_ray, :]) < 1e-4] = 0
    #     relative_loss = torch.linalg.vector_norm(relative_loss)
    #     return relative_loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.train_env:
            # consistancy
            with torch.no_grad():
                v01 = 0.5 * (self.env_map[:, 0, :] + self.env_map[:, -1, :])
                self.env_map[:, 0, :] = v01
                self.env_map[:, -1, :] = v01

            params = mi.traverse(self.scene)
            key = 'emitter.data'
            params[key] = self.env_map
            params.update()

    ############################ Validation ############################
    def validation_step(self, batch, batch_index):
        self.set_mode('val')
        spp = self.spp_validation

        # load data
        color_gt = batch['image_gt'][0]  # [3, imh, imw]
        alebdo_gt = batch['albedo_gt'][0]
        metadata = batch['metadata']
        imw = metadata['imw'][0]
        imh = metadata['imh'][0]

        sensor = create_mitsuba_sensor(metadata, spp)

        n_total_pixel = int(imw * imh)
        start_index = 0
        stop_index = 0
        n_pixel = self.n_pixel_val

        color_predict, Rd_predict, Rs_predict, rx_predict, ry_predict = [], [], [], [], []

        while (stop_index < n_total_pixel):
            ray, sampler, index, stop_index = generate_camera_rays_chunk(sensor, spp, n_pixel=n_pixel,
                                                                         start_index=start_index)
            color_predict_chunk, valid_ray, record = self(self.scene, sampler, ray)  # [n_pixel*spp, 3]


            # record
            valid_ray = valid_ray.torch().bool()

            # if valid_ray.shape[0] == 1:
            #     valid_ray = valid_ray.expand(color_predict_chunk.shape[0])  # in case all rays are unvalid
            color_predict_chunk[~valid_ray, :] = 1
            color_predict_chunk = torch.mean(color_predict_chunk.reshape(n_pixel, spp, 3),
                                             dim=1)  # [n_pixel, 3], average over spp
            color_predict.append(linear2srgb(color_predict_chunk).swapaxes(0, 1))  # [3, n_pixel]

            Rd_predict_chunk = record['Rd']  # [n_pixel*spp, 3]
            Rs_predict_chunk = record['Rs']  # [n_pixel*spp, 1]
            rx_predict_chunk = record['roughness_x']  # [n_pixel*spp, 1]

            Rd_predict_chunk[~valid_ray, :] = 1
            Rs_predict_chunk[~valid_ray, :] = 1
            rx_predict_chunk[~valid_ray, :] = 1

            Rd_predict.append(linear2srgb(torch.mean(Rd_predict_chunk.reshape(n_pixel, spp, 3), dim=1).swapaxes(0,
                                                                                                                1)))  # [3, n_pixel], average over spp
            Rs_predict.append(linear2srgb(torch.mean(Rs_predict_chunk.reshape(n_pixel, spp, 1), dim=1).swapaxes(0,
                                                                                                                1)))  # [1, n_pixel], average over spp
            rx_predict.append(linear2srgb(torch.mean(rx_predict_chunk.reshape(n_pixel, spp, 1), dim=1).swapaxes(0,
                                                                                                                1)))  # [1, n_pixel], average over spp

            # change index for the next interation
            start_index = stop_index
            if stop_index + n_pixel > n_total_pixel:
                n_pixel = n_total_pixel - stop_index

        # record
        color_predict = torch.concat(color_predict, dim=1).reshape(3, imh, imw)  # [3, imh, imw]
        Rd_predict = torch.concat(Rd_predict, dim=1).reshape(3, imh, imw)  # [3, imh, imw]
        Rs_predict = torch.concat(Rs_predict, dim=1).reshape(1, imh, imw)  # [1, imh, imw]
        rx_predict = torch.concat(rx_predict, dim=1).reshape(1, imh, imw)  # [1, imh, imw]

        # loss
        loss = self.mse(color_predict, color_gt)  # in sRGB space
        self.log('val_loss', loss, on_epoch=True)

        # all in sRGB space
        return {'val_loss': loss,
                'color_predict': color_predict,
                'color_gt': color_gt,
                'Rd_predict': Rd_predict,
                'Rd_gt': alebdo_gt,
                'Rs_predict': Rs_predict,
                'roughness_x_predict': rx_predict,
                }

    def validation_epoch_end(self, validation_step_outputs):

        # log images
        color_predict_list, color_gt_list, Rd_predict_list, Rd_gt_list, Rs_list, rx_list = [], [], [], [], [], []

        for i in validation_step_outputs:
            color_predict_list.append(self.ToPILImage(i['color_predict']))
            color_gt_list.append(self.ToPILImage(i['color_gt']))
            # Rd_predict_list.append(self.ToPILImage(linear2srgb(srgb2linear(i['Rd_predict']) * self.rgb_correction_linear)))
            Rd_predict_list.append(self.ToPILImage(i['Rd_predict']))
            Rd_gt_list.append(self.ToPILImage(i['Rd_gt']))
            Rs_list.append(self.ToPILImage(i['Rs_predict']))
            rx_list.append(self.ToPILImage(i['roughness_x_predict']))

        self.wandb_logger.log_image('image_pred', color_predict_list)
        self.wandb_logger.log_image('image_gt', color_gt_list)
        self.wandb_logger.log_image('Rd_pred', Rd_predict_list)
        self.wandb_logger.log_image('Rd_gt', Rd_gt_list)
        self.wandb_logger.log_image('Rs_pred', Rs_list)
        self.wandb_logger.log_image('roughness_x_pred', rx_list)

        # env_map
        env_map = self.env_map.clone().detach()[:, :, :3]  # HxWx3
        env_map = linear2srgb(torch.permute(env_map, (2, 0, 1)))  # 3xHxW
        self.wandb_logger.log_image('env_map', [self.ToPILImage(env_map)])

        env_map_gt = self.env_map_gt.clone().detach()[:, :, :3]  # HxWx3
        env_map_gt = linear2srgb(torch.permute(env_map_gt, (2, 0, 1)))  # 3xHxW
        self.wandb_logger.log_image('env_map_gt', [self.ToPILImage(env_map_gt)])

    @staticmethod
    def compute_abeldo_correction(albedo_gt, albedo_predict):
        """
        :param albedo_gt: masked gt albedo 3xHxW, torch, srgb
        :param albedo_predict: masked pred albedo 3xHxW, torch, srgb
        :return: scale, 3, torch, in linear space
        """
        albedo_gt = srgb2linear(albedo_gt)
        albedo_predict = srgb2linear(albedo_predict)
        albedo_gt = albedo_gt.reshape(3, -1)
        albedo_predict = albedo_predict.reshape(3, -1)
        scale = []
        for i in range(3):
            x_hat = albedo_predict[i, :]
            x = albedo_gt[i, :]
            _scale = x_hat.dot(x) / x_hat.dot(x_hat)
            scale.append(_scale)
        return torch.tensor(scale)

    ############################ Test ############################

    def on_test_start(self):
        # check mode
        assert self.mode not in ['train', 'val']
        # check log dir
        assert self.log_dir is not None

        params = mi.traverse(self.scene)
        key = 'emitter.data'
        # load relighting envmap, otherwise, use the envmap from training
        if self.mode == 'relighting':
            self.env_map = self.relighting_envmap
            key2 = 'emitter.scale'
            params[key2] = self.config.getfloat('testing', 'relighting_env_inten')
        params[key] = self.env_map
        params.update()

        self.spp_test = self.config.getint('testing', 'spp')
        self.n_pixel_test = self.config.getint('testing', 'n_pixel')

    def test_step(self, batch, batch_index):
        if batch_index in range(150, 200):
            spp = self.spp_test
            n_pixel = self.n_pixel_test

            # load data
            color_gt = batch['image_gt'][0]  # [3, imh, imw]
            alebdo_gt = batch['albedo_gt'][0]
            albedo_black = batch['albedo_gt_black'][0]
            metadata = batch['metadata']
            imw = metadata['imw'][0]
            imh = metadata['imh'][0]
            sensor = create_mitsuba_sensor(metadata, spp)

            n_total_pixel = int(imw * imh)
            start_index = 0
            stop_index = 0


            color_predict, Rd_predict, Rs_predict, rx_predict = torch.empty(3, 0).cuda(), torch.empty(3, 0).cuda(), torch.empty(1, 0).cuda(), torch.empty(1, 0).cuda()

            while (stop_index < n_total_pixel):
                ray, sampler, index, stop_index = generate_camera_rays_chunk(sensor, spp, n_pixel=n_pixel,
                                                                             start_index=start_index)
                color_predict_chunk, valid_ray, record = self(self.scene, sampler, ray)  # [n_pixel*spp, 3]

                # record
                valid_ray = valid_ray.torch().bool()
                # if valid_ray.shape[0] == 1:
                #     valid_ray = valid_ray.expand(color_predict_chunk.shape[0])  # in case all rays are unvalid
                color_predict_chunk[~valid_ray, :] = 1
                color_predict_chunk = torch.mean(color_predict_chunk.reshape(n_pixel, spp, 3),
                                                 dim=1)  # [n_pixel, 3], average over spp
                # color_predict.append(linear2srgb(color_predict_chunk).swapaxes(0, 1))  # [3, n_pixel]
                color_predict = torch.cat((color_predict, linear2srgb(color_predict_chunk).swapaxes(0, 1)), dim=1)

                Rd_predict_chunk = record['Rd']  # [n_pixel*spp, 3]
                if self.mode in ['test', 'relighting', 'material_editing']:
                    Rd_predict_chunk = Rd_predict_chunk * self.rgb_correction_linear.flatten()[None, :]


                Rs_predict_chunk = record['Rs']  # [n_pixel*spp, 1]
                rx_predict_chunk = record['roughness_x']  # [n_pixel*spp, 1]

                if self.mode == 'cal_correction':
                    # Rd_predict_chunk[~valid_ray, :] = 0  # black background
                    Rd_predict_chunk[~valid_ray, :] = 1  # white background  todo only for vis dtu
                else:
                    Rd_predict_chunk[~valid_ray, :] = 1
                Rs_predict_chunk[~valid_ray, :] = 1
                rx_predict_chunk[~valid_ray, :] = 1

                Rd_predict = torch.cat((Rd_predict, linear2srgb(torch.mean(Rd_predict_chunk.reshape(n_pixel, spp, 3), dim=1).swapaxes(0,1))), dim=1)
                Rs_predict = torch.cat((Rs_predict, linear2srgb(torch.mean(Rs_predict_chunk.reshape(n_pixel, spp, 1), dim=1).swapaxes(0,1))), dim=1)
                rx_predict = torch.cat((rx_predict, linear2srgb(torch.mean(rx_predict_chunk.reshape(n_pixel, spp, 1), dim=1).swapaxes(0,1))), dim=1)





                # change index for the next interation
                start_index = stop_index
                if stop_index + n_pixel > n_total_pixel:
                    n_pixel = n_total_pixel - stop_index

            color_predict = color_predict.reshape(3, imh, imw)  # [3, imh, imw]
            Rd_predict = Rd_predict.reshape(3, imh, imw)  # [3, imh, imw]
            Rs_predict = Rs_predict.reshape(1, imh, imw)  # [1, imh, imw]
            rx_predict = rx_predict.reshape(1, imh, imw)  # [1, imh, imw]

            # loss
            loss = self.mse(color_predict, color_gt)
            self.log('test_loss', loss, on_epoch=True)

            if self.mode == 'cal_correction':
                return {'color_predict': color_predict,
                        'color_gt': color_gt,
                        'Rd_predict': Rd_predict,
                        'Rd_gt': albedo_black,
                        'Rs_predict': Rs_predict,
                        'roughness_x_predict': rx_predict,
                        }
            else:
                image_pred = self.ToPILImage(color_predict)
                Rd_predict = self.ToPILImage(Rd_predict)
                Rs_predict = self.ToPILImage(Rs_predict)
                rx_predict = self.ToPILImage(rx_predict)
                color_gt = self.ToPILImage(color_gt)
                alebdo_gt = self.ToPILImage(alebdo_gt)

                self.wandb_logger.log_image('image_pred', [image_pred])
                self.wandb_logger.log_image('Rd_pred', [Rd_predict])
                self.wandb_logger.log_image('Rs_pred', [Rs_predict])
                self.wandb_logger.log_image('roughness_x_pred', [rx_predict])
                self.wandb_logger.log_image('image_gt', [color_gt])
                self.wandb_logger.log_image('Rd_gt', [alebdo_gt])

                log_dir = Path(self.log_dir, batch['data_id'][0])
                log_dir.mkdir(parents=True, exist_ok=True)

                if self.mode == 'test':
                    image_pred.save(Path(log_dir, 'rgb.png'))
                    Rd_predict.save(Path(log_dir, 'Rd.png'))
                    Rs_predict.save(Path(log_dir, 'Rs.png'))
                    rx_predict.save(Path(log_dir, 'roughness.png'))
                    color_gt.save(Path(log_dir, 'rgb_gt.png'))
                    alebdo_gt.save(Path(log_dir, 'Rd_gt.png'))
                if self.mode == 'relighting':
                    image_pred.save(Path(log_dir, f'{self.relighting_env_name}.png'))
                if self.mode == 'material_editing':
                    # image_pred.save(Path(log_dir, f"Rs_{self.config.getfloat('testing', 'Rs')}_roughness_{self.config.getfloat('testing', 'roughness')}.png"))
                    image_pred.save(Path(log_dir, f"me.png"))
                return 0

    def test_epoch_end(self, test_step_outputs):

        if self.mode == 'cal_correction':
            # compute correction
            if not self.train_env:
                correction = torch.ones(3)
            else:
                n_correction = min(len(test_step_outputs), 3)
                correction = []
                for i in range(n_correction):
                    correction.append(self.compute_abeldo_correction(test_step_outputs[i]['Rd_gt'],
                                                                     test_step_outputs[i][
                                                                         'Rd_predict']))
                correction = torch.stack(correction, dim=0).mean(dim=0)

            self.rgb_correction_linear = correction[:, None, None].cuda()

            print('\n')
            print('*' * 20)
            print('correction')
            print(f"R: {self.rgb_correction_linear.flatten()[0]}, G: {self.rgb_correction_linear.flatten()[1]}, B: {self.rgb_correction_linear.flatten()[2]}")
            print('*' * 20)

            # record
            color_predict_list, color_gt_list, Rd_predict_list, Rd_gt_list, Rs_list, rx_list, ry_list = [], [], [], [], [], [], []
            for i in test_step_outputs:
                color_predict_list.append(self.ToPILImage(i['color_predict']))
                color_gt_list.append(self.ToPILImage(i['color_gt']))
                Rd_predict_list.append(self.ToPILImage(linear2srgb(srgb2linear(i['Rd_predict']) * self.rgb_correction_linear)))
                Rd_gt_list.append(self.ToPILImage(i['Rd_gt']))
                Rs_list.append(self.ToPILImage(i['Rs_predict']))
                rx_list.append(self.ToPILImage(i['roughness_x_predict']))

            self.wandb_logger.log_image('image_pred', color_predict_list)
            self.wandb_logger.log_image('image_gt', color_gt_list)
            self.wandb_logger.log_image('Rd_pred', Rd_predict_list)
            self.wandb_logger.log_image('Rd_gt', Rd_gt_list)
            self.wandb_logger.log_image('Rs_pred', Rs_list)
            self.wandb_logger.log_image('roughness_x_pred', rx_list)

        # env_map
        env_map = self.env_map.clone().detach()[:, :, :3]  # HxWx3
        env_map = linear2srgb(torch.permute(env_map, (2, 0, 1)) / self.rgb_correction_linear)  # 3xHxW
        # env_map = linear2srgb(torch.permute(env_map, (2, 0, 1)))  # 3xHxW
        self.wandb_logger.log_image('env_map', [self.ToPILImage(env_map)])

        env_map_gt = self.env_map_gt.clone().detach()[:, :, :3]  # HxWx3
        env_map_gt = linear2srgb(torch.permute(env_map_gt, (2, 0, 1)))  # 3xHxW
        self.wandb_logger.log_image('env_map_gt', [self.ToPILImage(env_map_gt)])

        if self.mode == 'test':
            self.ToPILImage(env_map).save(Path(self.log_dir, 'env_map.png'))
            self.ToPILImage(env_map_gt).save(Path(self.log_dir, 'env_map_gt.png'))


    def on_save_checkpoint(self, checkpoint):
        # register RGB correction as part of the checkpoint
        checkpoint['rgb_correction_linear'] = self.rgb_correction_linear

    def on_load_checkpoint(self, checkpoint):
        self.rgb_correction_linear = checkpoint['rgb_correction_linear']


