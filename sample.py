import argparse, os, sys, glob
import torch
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from main import DataModuleFromConfig


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def save_nifti(img, path):
    img = img.squeeze(0)  # remove batch dimension, now it's (1, 192, 192, 160)
    if len(img.shape) != 4: return
    img = img.permute(1, 2, 3, 0)  # reorder dimensions to be compatible with nibabel

    img = img.numpy()

    os.makedirs(os.path.split(path)[0], exist_ok=True)

    nifti_img = nib.Nifti1Image(img, np.eye(4))  # you might want to replace np.eye(4) with the correct affine matrix
    nib.save(nifti_img, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--base",
        type=str,
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default='configs/latent-diffusion/brats-ldm-vq-4.yaml',
    )

    parser.add_argument(
        "--source",
        type=str,
        nargs="?",
        default="dense",
        help="the source modality (select dense)",
    )

    parser.add_argument(
        "--target",
        type=str,
        nargs="?",
        default="cine",
        help="the target modality (select cine)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default= 500,  #200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=48,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=48,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--D",
        type=int,
        default=20,
        help="image depth, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    config = OmegaConf.load(opt.base)  # TODO: Optionally download from same location as ckpt and chnage this logic
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    data = data.datasets["test"]
    
    model = load_model_from_config(config, config.model.params.ckpt_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    #modalities = ['t1', 't1ce', 't2', 'flair']
    modalities = ['dense', 'cine']

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    counter = 1
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            for batch in tqdm(data, desc="Data"):
                #subject_id = batch["subject_id"]
                x_src = batch[opt.source].unsqueeze(0).to(device)
                # z_src , _, _ = model.first_stage_model.encode(x_src)
                # z_tgtl, _, _ = model.first_stage_model.encode(x_src, opt.target)
                # z_src = model.get_first_stage_encoding(z_src).detach()
                # z_tgtl = model.get_first_stage_encoding(z_tgtl).detach()

                z_src = x_src
                z_tgtl = x_src

                z_src = torch.cat([z_src, z_tgtl], dim=1)
            
                x0 = z_src
                c = modalities.index(opt.target)
                c = torch.nn.functional.one_hot(torch.tensor(c), num_classes=4).float()
                c = c.unsqueeze(0).repeat(z_src.shape[0], 1).unsqueeze(1).to(device)
                shape = [6, opt.H//4, opt.W//4, opt.D//4]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 x0=x0,
                                                 eta=opt.ddim_eta,
                                                 counter=counter)

                #x_samples_ddim = model.decode_first_stage(samples_ddim)
                #x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

                #x_samples_ddim = torch.clamp((samples_ddim+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

                # for x_sample in x_samples_ddim:
                #     save_nifti(x_sample, os.path.join(sample_path, os.path.join( f"{base_count:04}.nii.gz")))
                #     base_count += 1
                
                #x_samples_ddim = samples_ddim
                #save_nifti(x_samples_ddim, os.path.join(sample_path, f"{subject_id}_{opt.source}_to_{opt.target}.nii.gz"))
                #save_nifti(x_samples_ddim, os.path.join(sample_path, f"{counter}_{opt.source}_to_{opt.target}.nii.gz"))
                counter += 1
    import cv2
    import os
    import re

    def get_time_index(filename):
        match = re.search(r'time_(\d+)', filename)
        if match:
            return int(match.group(1))
        return -1  # Return -1 if no match found

    def create_video_from_images(input_folder, output_file, fps=10):
        # Get all image files
        images = [img for img in os.listdir(input_folder) if img.startswith("time_") and (img.endswith(".png") or img.endswith(".jpg"))]
        
        # Sort images based on the time index
        images.sort(key=lambda x: get_time_index(x))

        if not images:
            print(f"No images found in {input_folder}")
            return

        # Read the first image to get dimensions
        frame = cv2.imread(os.path.join(input_folder, images[0]))
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for image in images:
            img_path = os.path.join(input_folder, image)
            frame = cv2.imread(img_path)
            out.write(frame)
            print(f"Processed: {image}")

        out.release()
        print(f"Video created: {output_file}")

    def process_folders(base_folder):
        for subfolder in ['1', '2']:
            input_folder = os.path.join(base_folder, subfolder)
            output_file = os.path.join(base_folder, f'video_{subfolder}.mp4')
            create_video_from_images(input_folder, output_file)

    # Usage
    base_folder = 'Synthesized Cine through time'
    process_folders(base_folder)

    import target_cine_video

    # import os
    # import numpy as np
    # import nibabel as nib
    # import cv2

    # def extract_frame_from_nifti(file_path):
    #     # Load the NIfTI file
    #     img = nib.load(file_path)
    #     data = img.get_fdata()
        
    #     # Extract the 0th frame of the last index (assuming shape is 48,48,20)
    #     frame = data[:,:,0]
        
    #     # Normalize to 0-255 range for video
    #     frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
    #     return frame

    # def create_video_from_years(base_folder, start_year, end_year, output_file, fps=10):
    #     frames = []
    #     for year in range(start_year, end_year + 1):
    #         folder = os.path.join(base_folder, str(year))
    #         nifti_file = os.path.join(folder, str(year)+'_cine.nii.gz')
            
    #         if os.path.exists(nifti_file):
    #             frame = extract_frame_from_nifti(nifti_file)
    #             frames.append(frame)
    #             print(f"Processed: {nifti_file}")
    #         else:
    #             print(f"File not found: {nifti_file}")
        
    #     if not frames:
    #         print(f"No frames extracted for years {start_year}-{end_year}")
    #         return
        
    #     # Create video
    #     height, width = frames[0].shape
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
    #     for frame in frames:
    #         # Convert grayscale to RGB
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #         out.write(rgb_frame)
        
    #     out.release()
    #     print(f"Video created: {output_file}")

    # def process_year_ranges(base_folder):
    #     create_video_from_years(base_folder, 1960, 1979, os.path.join(base_folder, 'video_1960_1979.mp4'))
    #     create_video_from_years(base_folder, 1980, 1999, os.path.join(base_folder, 'video_1980_1999.mp4'))

    # # Usage
    # base_folder = 'Nifty/ValidationData'
    # process_year_ranges(base_folder)

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")