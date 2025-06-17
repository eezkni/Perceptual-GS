#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import torch
import torch.nn as nn
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_imp, render_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB
import uuid
import random
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from time import time
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, no_hd, no_md, edge_mode, no_sadr, no_ada_densification_strategy, all_dr, no_od, od_factor):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, edge_mode=edge_mode, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    bce_loss = nn.BCELoss()

    total_time = 0

    for iteration in range(first_iter, opt.iterations + 1):    
        start = time()    
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_sens = 0

        if iteration <= opt.densify_until_iter:
            r_sens = render(viewpoint_cam, gaussians, pipe, bg, override_color=(1-gaussians.get_sensitivity).repeat(1, 3))["render"][0]
            r_sens = r_sens[None, :]

            gt_sens = viewpoint_cam.sens.cuda()

            loss_sens = bce_loss(1-r_sens, gt_sens)

            loss = (1.0 - opt.lambda_bce) * loss + opt.lambda_bce * loss_sens

        try:
            gt_alpha = viewpoint_cam.alpha.cuda()
        except:
            gt_alpha = None

        if gt_alpha != None:
            r_alpha = render(viewpoint_cam, gaussians, pipe, bg, override_color=gaussians.get_alpha.repeat(1, 3))["render"]
            loss_alpha = l1_loss(r_alpha, gt_alpha)

            loss = 0.9*loss + 0.1 * loss_alpha

        loss.backward()

        if iteration <= opt.densify_until_iter and (not no_hd or not no_md):
            if iteration % opt.hd_interval == 0 or iteration % opt.md_interval == 0:
                with torch.no_grad():
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    views = scene.getTrainCameras()

                    for view in views:
                        torch.cuda.empty_cache()
                        render_pkg = render_imp(view, gaussians, pipe, bg)
                        acc_weight = render_pkg["accum_weights"]
                        gaussians.trans = torch.max(acc_weight.reshape(-1,1), gaussians.trans)
        torch.cuda.empty_cache()
        
        iter_end.record()
        end = time()
        total_time += (end-start)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration == 600:
                scale = torch.max(gaussians.get_scaling, dim=1).values.reshape(-1, 1)
                k = int(scale.size(0) * 0.25)
                threshold = torch.topk(scale.squeeze(), k, largest=True).values.min()
                mask = (scale >= threshold) 
                masked_sens = gaussians.get_sensitivity * mask.reshape(-1, 1)
                mix_rate = (torch.logical_and(masked_sens>opt.threshold_low, masked_sens<opt.threshold_high)).count_nonzero() / masked_sens.count_nonzero()
                gaussians.sparse = mix_rate > 0.55

                avg = 0

                views = scene.getTrainCameras()

                for view in views:
                    torch.cuda.empty_cache()
                    edge = torch.count_nonzero(view.sens.cuda()>0)
                    num = torch.count_nonzero(view.sens.cuda()<=1)

                    avg += edge/num
                
                avg = avg/len(views)

                gaussians.split_only = avg > 0.85

                if all_dr:
                    gaussians.sparse = True

                if no_ada_densification_strategy:
                    gaussians.split_only = True

                print("Split Only: ", gaussians.split_only)
                print("Sparse: ", gaussians.sparse)
                print("Mode: ", edge_mode)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if not no_hd and not no_md and gaussians.sparse and iteration % 5000 == 0:
                        pass
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, iteration, opt, no_hd, no_md, no_od, od_factor)
                
                if not no_hd and not no_md and gaussians.sparse and iteration%5000==0:
                    out_pts_list=[]
                    gt_list=[]
                    views=scene.getTrainCameras()
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        render_depth_pkg = render_depth(view, gaussians, pipe, background)
                        out_pts = render_depth_pkg["out_pts"]
                        accum_alpha = render_depth_pkg["accum_alpha"]

                        prob=1-accum_alpha

                        prob = prob/prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()

                        factor=1/(image.shape[1]*image.shape[2]*len(views)/3_500_000)

                        N_xyz=prob.shape[0]
                        num_sampled=int(N_xyz*factor)

                        indices = np.random.choice(N_xyz, size=num_sampled, p=prob)
                        indices=np.unique(indices)
                        
                        out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                        gt = gt.permute(1,2,0).reshape(-1,3)

                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])       

                    out_pts_merged=torch.cat(out_pts_list)
                    gt_merged=torch.cat(gt_list)

                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    viewpoint_stack = scene.getTrainCameras().copy()

                if not gaussians.sparse or no_hd or no_md:
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                torch.cuda.empty_cache()

            if not no_hd and not no_md and gaussians.sparse and iteration == opt.densify_until_iter:

                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.reinitial_pts(gaussians._xyz, 
                                    SH2RGB(gaussians._features_dc+0)[:,0])
                
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                viewpoint_stack = scene.getTrainCameras().copy()

            if gt_alpha != None and iteration > 1000 and iteration % 100 == 0:
                prune_mask = (gaussians.get_alpha < 0.7).squeeze()
                gaussians.prune_points(prune_mask)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print("**********************************")
    print("Scene:", dataset.source_path)
    print("Gaussians:", gaussians._xyz.shape[0])

def prepare_output_and_logger(args):    
    dataset = args.source_path.split("/")[-2]
    scene = args.source_path.split("/")[-1]

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            # unique_str = str(uuid.uuid4())
            unique_str = dataset + "/" + scene
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # sensitivity test
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        bce_loss = nn.BCELoss()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                bce_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    sens = torch.clamp(render(viewpoint, scene.gaussians, renderArgs[0], renderArgs[1], override_color=(scene.gaussians.get_sensitivity).repeat(1, 3))["render"][0], 0.0, 1.0)
                    sens = sens[None, :]
                    gt_sens = torch.clamp(viewpoint.sens.to("cuda"), 0.0, 1.0)

                    bce_test += bce_loss(sens, gt_sens).mean().double()
                    psnr_test += psnr(sens, gt_sens).mean().double()
                psnr_test /= len(config['cameras'])
                bce_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: BCE {} PSNR {}".format(iteration, config['name'], bce_test, psnr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=randint(1000, 9999))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_hd", action="store_true")
    parser.add_argument("--no_md", action="store_true")
    parser.add_argument("--no_sadr", action="store_true")
    parser.add_argument("--no_ada_densification_strategy", action="store_true")
    parser.add_argument("--all_dr", action="store_true")
    parser.add_argument("--no_od", action="store_true")
    parser.add_argument("--edge_mode", type=str, default = "sensitivity_maps")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.no_hd, args.no_md, args.edge_mode, args.no_sadr, args.no_ada_densification_strategy, args.all_dr, args.no_od, args.od_factor)

    # All done
    print("\nTraining complete.")
