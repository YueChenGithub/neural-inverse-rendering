import mitsuba as mi
import numpy as np
import math
import torch
import drjit as dr

mi.set_variant("cuda_ad_rgb")


def generate_camera_rays(sensor, spp):
    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_sample_count = dr.prod(film_size) * spp
    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)  # todo look into the seed
        # sampler.seed(torch.randint(0, 10000, (1,)).item(), total_sample_count)
    pos = dr.arange(mi.UInt32, total_sample_count)
    pos //= spp
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(pos % int(film_size[0])),
                      mi.Float(pos // int(film_size[0])))  # [[p1]*spp, [p2]*spp, ...]
    pos += sampler.next_2d()
    rays, weights = sensor.sample_ray_differential(
        time=0,
        sample1=0,
        # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
        sample2=pos * scale,
        # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
        sample3=0
        # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
    )

    return rays, sampler


def create_mitsuba_sensor(metadata, spp):
    # read metadata, if metadata is a list, only use the first one (batchsize=1)
    if isinstance(metadata['cam_transform_mat'], list):
        cam_transform_mat = metadata['cam_transform_mat'][0]
        cam_angle_x = metadata['cam_angle_x'][0]
        imw = metadata['imw'][0]
        imh = metadata['imh'][0]
    else:
        cam_transform_mat = metadata['cam_transform_mat']
        cam_angle_x = metadata['cam_angle_x']
        imw = metadata['imw']
        imh = metadata['imh']

    cam_transform_mat = np.array(list(cam_transform_mat.split(',')), dtype=float)  # str2array
    cam_transform_mat = cam_transform_mat.reshape(4, 4)

    cam_transform_mat = mi.ScalarTransform4f(cam_transform_mat) @ mi.ScalarTransform4f.scale(
        [-1, 1, -1])  # change coordinate from blender to mitsuba (flip x and z axis)
    # cam_transform_mat = mi.ScalarTransform4f(cam_transform_mat)

    sensor_dict = {'type': 'perspective',
                   'to_world': cam_transform_mat,
                   'fov': float(cam_angle_x * 180 / math.pi),
                   'film': {'type': 'hdrfilm',
                            'width': int(imw),
                            'height': int(imh),
                            # # Use a box reconstruction filter
                            # 'filter': {'type': 'box'}
                            },
                   'sampler': {'type': 'stratified',  # 'independent',
                               'sample_count': spp
                               },
                   # 'principal_point_offset_x': -7.42566406 / 384,
                   # 'principal_point_offset_y': -6.10269531 / 384,  # todo!! onfy for dtu

                   }

    # sensor_dict = {'type': 'orthographic',
    #                'to_world': cam_transform_mat,
    #                'film': {'type': 'hdrfilm',
    #                         'width': int(imw),
    #                         'height': int(imh),
    #                         # # Use a box reconstruction filter
    #                         # 'filter': {'type': 'box'}
    #                         },
    #                'sampler': {'type': 'stratified',  # 'independent',
    #                            'sample_count': spp
    #                            }
    #                }

    sensor = mi.load_dict(sensor_dict)

    return sensor


def create_mitsuba_scene_envmap(mesh_path, envmap_path, inten):
    envmap_dict = {'type': 'envmap',
                   'filename': envmap_path,
                   'scale': inten,
                   'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                               mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}
                               # mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=-90)}  # todo only for dtu

    object_dict = {'type': 'obj',
                   'filename': mesh_path,
                   'face_normals': True,  # todo
                   }

    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


@torch.no_grad()
def PowerHeuristic(fPdf, gPdf):
    """ Compute the weight for multiple importance sampling

    Args:
        fPdf: pdf for f
        gPdf: pdf for g

    Returns:

    """
    return (fPdf * fPdf) / (fPdf * fPdf + gPdf * gPdf).clamp_min(1e-4)


def m2t(a, unsqueeze=True):
    """ Convert Mitsuba tensor to PyTorch tensor

    Args:
        a: mitsuba tensor
        unsqueeze: add a dimension to the tensor?

    Returns:

    """
    if unsqueeze:
        a = a.torch().unsqueeze(1)
        a[a.isnan()] = 0
        return a
    else:
        a = a.torch()
        a[a.isnan()] = 0
        return a


def generate_camera_rays_randomly(sensor, spp, n_pixel=1024 * 8):
    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()

    index = np.random.choice(dr.prod(film_size), n_pixel, replace=False)
    total_sample_count = n_pixel * spp

    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)  # todo look into the seed
        # sampler.seed(torch.randint(0, 10000, (1,)).item(), total_sample_count)
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(np.array(index % int(film_size[0])).repeat(spp)),
                      mi.Float(np.array(index // int(film_size[0])).repeat(spp)))
    pos += sampler.next_2d()
    rays, weights = sensor.sample_ray_differential(
        time=0,
        sample1=0,
        # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
        sample2=pos * scale,
        # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
        sample3=0
        # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
    )

    return rays, sampler, index


def generate_camera_rays_chunk(sensor, spp, n_pixel=1024 * 8, start_index=0):
    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()

    stop_index = start_index + n_pixel
    index = np.arange(start=start_index, stop=stop_index)
    total_sample_count = n_pixel * spp

    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)  # todo look into the seed
        # sampler.seed(torch.randint(0, 10000, (1,)).item(), total_sample_count)
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(np.array(index % int(film_size[0])).repeat(spp)),
                      mi.Float(np.array(index // int(film_size[0])).repeat(spp)))
    pos += sampler.next_2d()
    rays, weights = sensor.sample_ray_differential(
        time=0,
        sample1=0,
        # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
        sample2=pos * scale,
        # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
        sample3=0
        # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
    )

    return rays, sampler, index, stop_index
