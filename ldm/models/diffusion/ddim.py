"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib as plt
import os 
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        count = kwargs['count']
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    count=count,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, count=None):
        device = self.model.betas.device
        b = shape[0]
        #img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *x0.shape[2:])).to(device)], dim=1)

        
        # import nibabel as nib
        # source_img = img[0,2,...].cpu().numpy()
    
        # # Remove singleton dimensions and ensure correct orientation
        # tensor_numpy = np.squeeze(source_img)
        
        # # NiBabel expects data in (x, y, z) order, so we might need to transpose
        # # Assuming the tensor is in (depth, height, width) order:
        # #tensor_numpy = np.transpose(tensor_numpy, (2, 1, 0))
        
        # # Create a NIfTI image
        # nifti_image = nib.Nifti1Image(tensor_numpy, affine=np.eye(4))
        # nib.save(nifti_image, 'sample_noisy.nii.gz')

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_prevs = []
        import os
        save_dir = os.path.join(os.getcwd(), 'ddim_sampling_progress/'+str(count))
        os.makedirs(save_dir, exist_ok=True)
        save_dir_t = os.path.join(os.getcwd(), 'Synthesized Cine through time/'+str(count))
        os.makedirs(save_dir_t, exist_ok=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            x_prev, pred_x0 = outs
            x_prevs.append(x_prev.cpu())
            self.save_image(x_prev, save_dir, f'step_{step}.png')
            img = torch.cat([x0, x_prev], dim=1)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
            
        for t in range(x_prev.shape[-1]):
            self.save_image_time(x_prev[0,0,:,:,t].cpu().numpy(), save_dir_t, f'time_{t}.png')
        
        import nibabel as nib
        import os
        demo_dense = img[:, -x_prev.shape[1]:]
        #demo_dense = img[:, 0]
        source_img = demo_dense[0,...].cpu().numpy()
        tensor_numpy = np.squeeze(source_img)
        nifti_image = nib.Nifti1Image(tensor_numpy, affine=np.eye(4))
        nib.save(nifti_image, os.path.join('outputs/samples', f"{i}.nii.gz"))
        return img[:, -x_prev.shape[1]:], intermediates
        #return img[:, 0], intermediates

        # import nibabel as nib
        # demo_dense = img[:,-1]
        # source_img = demo_dense[0,...].cpu().numpy()
    
        # # Remove singleton dimensions and ensure correct orientation
        # tensor_numpy = np.squeeze(source_img)
        
        # # NiBabel expects data in (x, y, z) order, so we might need to transpose
        # # Assuming the tensor is in (depth, height, width) order:
        # #tensor_numpy = np.transpose(tensor_numpy, (2, 1, 0))
        
        # # Create a NIfTI image
        # nifti_image = nib.Nifti1Image(tensor_numpy, affine=np.eye(4))
        # nib.save(nifti_image, 'x3.nii.gz')

    def save_image(self, tensor, save_dir, filename):
        import matplotlib as plt
        from PIL import Image
        # tensor shape is (1, 1, 48, 48, 20)
        # We'll save the (48, 48) image from the first channel and the first slice of the last dimension
        img = tensor[0, 0, :, :, 0].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8) 

        im = Image.fromarray(img)
        save_path = os.path.join(save_dir, filename)
        im.save(save_path)

    def save_image_time(self, tensor, save_dir, filename):
        import matplotlib as plt
        from PIL import Image
        # tensor shape is (1, 1, 48, 48, 20)
        # We'll save the (48, 48) image from the first channel and the first slice of the last dimension
        #img = tensor[0, 0, :, :, 0].cpu().numpy()
        img = tensor
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8) 

        im = Image.fromarray(img)
        save_path = os.path.join(save_dir, filename)
        im.save(save_path)
        

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        #pred_x0 = (x[:,-e_t.shape[1]:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_x0 = (x[:,-e_t.shape[1]:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        #temperature = 0.0001
        noise = sigma_t * noise_like(e_t.shape, device, repeat_noise) * temperature
        #noise = sigma_t * noise_like(e_t.shape, device, repeat_noise)
        # if noise_dropout > 0.:
        #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        #x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt 
        return x_prev, pred_x0
