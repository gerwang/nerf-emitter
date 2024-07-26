import gc
from contextlib import contextmanager

import drjit as dr
import mitsuba as mi
import torch

from nerfstudio.models.base_model import Model


def clear_memory():
    gc.collect()
    gc.collect()
    dr.flush_malloc_cache()
    torch.cuda.empty_cache()


denoiser = None
denoiser_size = (0, 0)


def render_aggregate(scene, spp, spp_per_batch, seed=0, seed_grad=None, power_of_two=False, denoise=False,
                     denoise_no_gbuffer=False, **kwargs):
    iter_spps = divide_spp(spp, spp_per_batch, power_of_two)
    if seed_grad is None:
        seed_grad = seed + len(iter_spps)
    img_sum = None
    img = None
    for i, iter_spp in enumerate(iter_spps):
        with dr.suspend_grad(when=i < len(iter_spps) - 1):
            img = mi.render(scene, **kwargs, spp=iter_spp, spp_grad=iter_spp,
                            seed=seed + i, seed_grad=seed_grad + i)
        if denoise:
            assert 'sensor' in kwargs
            sensor = kwargs['sensor']
            img = sensor.film().bitmap()
            global denoiser, denoiser_size
            if denoiser is None or tuple(img.size()) != denoiser_size:
                denoiser = mi.OptixDenoiser(input_size=img.size(), albedo=True, normals=True, temporal=False)
                denoiser_size = tuple(img.size())
            img = denoiser(img, albedo_ch="albedo", normals_ch="normals", to_sensor=sensor.world_transform().inverse())
            img = mi.TensorXf(img)
        if img_sum is None:
            img_sum = dr.detach(img)
        else:
            img_sum += dr.detach(img)
    # gradient tricks
    img = img - dr.detach(img) + img_sum / len(iter_spps)

    this_denoiser_size = (img.shape[1], img.shape[0])
    if denoise_no_gbuffer:
        if denoiser is None or this_denoiser_size != denoiser_size:
            denoiser = mi.OptixDenoiser(input_size=this_denoiser_size)
            denoiser_size = this_denoiser_size
        img = denoiser(img)
    return img


def forward_grad_aggregate(scene, param, spp, spp_per_batch, set_value, seed=0, seed_grad=None, power_of_two=False,
                           **kwargs):
    iter_spps = divide_spp(spp, spp_per_batch, power_of_two)
    img_sum = None
    grad_sum = None
    if seed_grad is None:
        seed_grad = seed + len(iter_spps)
    for i, iter_spp in enumerate(iter_spps):
        set_value()
        dr.forward(param, dr.ADFlag.ClearEdges)
        img = mi.render(scene, **kwargs, spp=iter_spp, seed=seed + i, seed_grad=seed_grad + i)
        dr.forward_to(img)
        grad = dr.grad(img)
        dr.eval(grad)
        if img_sum is None:
            img_sum = img
        else:
            img_sum += img
        if grad_sum is None:
            grad_sum = grad
        else:
            grad_sum += grad
    return img_sum / len(iter_spps), grad_sum / len(iter_spps)


def divide_spp(spp, spp_per_batch, power_of_two=False):
    cnt, rem = divmod(spp, spp_per_batch)
    res = []
    if power_of_two and cnt > 1:
        cnt -= 1
        # divide first spp_per_batch into power of twos, 1, 1, 2, 4, 8, 16, ...
        first = True
        current = 1
        remaining = spp_per_batch
        while remaining > 0:
            share = min(current, remaining)
            res.append(share)
            remaining -= share
            if first:
                first = False
            else:
                current *= 2
    res.extend([spp_per_batch for _ in range(cnt)])
    if rem != 0:
        res.append(rem)
    return res


@contextmanager
def disable_aabb(model: Model, when=True):
    if when:
        model.set_disable_aabb(True)
    try:
        yield
    finally:
        if when:
            model.set_disable_aabb(False)
