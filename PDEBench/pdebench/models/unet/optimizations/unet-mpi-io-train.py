from __future__ import annotations

import logging
import pickle
import os
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn

from pdebench.models.metrics import metrics
from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetMult, UNetDatasetSingle

# Configure logging to output only on rank 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initializes the distributed process group and device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback for non-distributed runs
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Not running in distributed mode.")
    
    return device, rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def run_training(
    if_training,
    continue_training,
    num_workers,
    initial_step,
    t_train,
    in_channels,
    out_channels,
    batch_size,
    unroll_step,
    ar_mode,
    pushforward,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    model_update,
    flnm,
    single_file,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    base_path="../data/",
    training_type="autoregressive",
):
    # 1. Setup Distributed Environment
    device, rank, local_rank, world_size = setup_distributed()

    # Only log on rank 0
    if rank == 0:
        msg = f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}, World Size={world_size}"
        logger.info(msg)

    ################################################################
    # load data
    ################################################################

    if single_file:
        # filename
        model_name = flnm[:-5] + "_Unet"

        # Initialize the dataset
        train_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
        )
        val_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
        )

    else:
        # filename
        model_name = flnm + "_Unet"

        train_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            saved_folder=base_path,
        )
        val_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
            saved_folder=base_path,
        )

    # 2. Setup Distributed Samplers
    # Ensures each rank gets a different slice of the data
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)

    # 3. Update DataLoaders
    # shuffle must be False because the sampler handles shuffling
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True
    )

    ################################################################
    # training and evaluation
    ################################################################

    # Get sample to determine dimensions (ensure sync isn't needed here, assuming data is uniform)
    _, _data = next(iter(val_loader))
    dimensions = len(_data.shape)
    
    if rank == 0:
        msg = f"Spatial Dimension: {dimensions - 3}"
        logger.info(msg)

    # Initialize Model
    if training_type in ["autoregressive"]:
        if dimensions == 4:
            model = UNet1d(in_channels * initial_step, out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(in_channels * initial_step, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels * initial_step, out_channels).to(device)
    if training_type in ["single"]:
        if dimensions == 4:
            model = UNet1d(in_channels, out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(in_channels, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels, out_channels).to(device)

    # 4. Wrap Model in DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Set maximum time step of the data to train
    t_train = min(t_train, _data.shape[-2])
    
    if t_train - unroll_step < 1:
        unroll_step = t_train - 1

    if training_type in ["autoregressive"]:
        if ar_mode:
            if pushforward:
                model_name = model_name
            if not pushforward:
                unroll_step = _data.shape[-2]
                model_name = model_name + "-AR"
        else:
            model_name = model_name + "-1-step"

    model_path = model_name + ".pt"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        msg = f"Total parameters = {total_params}"
        logger.info(msg)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.inf

    start_epoch = 0

    if not if_training:
        # Load checkpoint (map to local device to avoid GPU OOM on rank 0)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle state dict keys (DDP adds 'module.' prefix)
        state_dict = checkpoint["model_state_dict"]
        if dist.is_initialized():
             # If loading a non-DDP checkpoint into DDP model
            if not list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Metrics usually run only on rank 0 or need aggregation. 
        # For simplicity, we run on rank 0 here.
        if rank == 0:
            Lx, Ly, Lz = 1.0, 1.0, 1.0
            errs = metrics(
                val_loader,
                model.module if dist.is_initialized() else model, # unwrapped for metrics if needed
                Lx, Ly, Lz,
                plot, channel_plot,
                model_name,
                x_min, x_max, y_min, y_max, t_min, t_max,
                mode="Unet",
                initial_step=initial_step,
            )
            pickle.dump(errs, Path.open(model_name + ".pickle", "wb"))
        
        cleanup_distributed()
        return

    if continue_training:
        if rank == 0:
            msg = "Restoring model (that is the network's weights) from file..."
            logger.info(msg)
            
        checkpoint = torch.load(model_path, map_location=device)
        
        state_dict = checkpoint["model_state_dict"]
        # Compatibility fix for DDP/Non-DDP loading
        if dist.is_initialized() and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.train()

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]

    if rank == 0:
        msg = "start training..."
        logger.info(msg)

    # ---------------- TRAINING LOOP ----------------
    if ar_mode:
        for ep in range(start_epoch, epochs):
            # 5. Set Epoch for Sampler (Crucial for shuffling)
            train_sampler.set_epoch(ep)
            
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0
                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)

                if training_type in ["autoregressive"]:
                    pred = yy_tensor[..., :initial_step, :]
                    inp_shape = list(xx_tensor.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    for t in range(initial_step, t_train):
                        if t < t_train - unroll_step:
                            with torch.no_grad():
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)
                                y = yy_tensor[..., t : t + 1, :]
                                temp_shape = [0]
                                temp_shape.extend(list(range(2, len(inp.shape))))
                                temp_shape.append(1)
                                im = model(inp).permute(temp_shape).unsqueeze(-2)
                                pred = torch.cat((pred, im), -2)
                                xx_tensor = torch.cat((xx_tensor[..., 1:, :], im), dim=-2)
                        else:
                            inp = xx_tensor.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)
                            y = yy_tensor[..., t : t + 1, :]
                            temp_shape = [0]
                            temp_shape.extend(list(range(2, len(inp.shape))))
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2)
                            loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                            pred = torch.cat((pred, im), -2)
                            xx_tensor = torch.cat((xx_tensor[..., 1:, :], im), dim=-2)

                    train_l2_step += loss.item()
                    _batch = yy_tensor.size(0)
                    _yy = yy_tensor[..., :t_train, :]
                    l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                    train_l2_full += l2_full.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if training_type in ["single"]:
                x = xx[..., 0, :]
                y = yy[..., t_train - 1 : t_train, :]
                pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
                _batch = yy.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                train_l2_step += loss.item()
                train_l2_full += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % model_update == 0:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)

                        if training_type in ["autoregressive"]:
                            pred = yy_tensor[..., :initial_step, :]
                            inp_shape = list(xx.shape)
                            inp_shape = inp_shape[:-2]
                            inp_shape.append(-1)
                            for t in range(initial_step, t_train):
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)
                                y = yy_tensor[..., t : t + 1, :]
                                temp_shape = [0]
                                temp_shape.extend(list(range(2, len(inp.shape))))
                                temp_shape.append(1)
                                im = model(inp).permute(temp_shape).unsqueeze(-2)
                                loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                                pred = torch.cat((pred, im), -2)
                                xx_tensor = torch.cat((xx_tensor[..., 1:, :], im), dim=-2)

                            val_l2_step += loss.item()
                            _batch = yy.size(0)
                            _pred = pred[..., initial_step:t_train, :]
                            _yy = yy_tensor[..., initial_step:t_train, :]
                            val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                    if training_type in ["single"]:
                        x = xx[..., 0, :]
                        y = yy[..., t_train - 1 : t_train, :]
                        pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
                        val_l2_step += loss.item()
                        val_l2_full += loss.item()

                    # Aggregate validation loss across all ranks for accurate best-model saving
                    if dist.is_initialized():
                        val_loss_tensor = torch.tensor([val_l2_full], device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_l2_full = val_loss_tensor.item()

                    # 6. Save Model Only on Rank 0
                    if rank == 0 and val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        torch.save(
                            {
                                "epoch": ep,
                                # Save the underlying model, not the DDP wrapper
                                "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_val_min,
                            },
                            model_path,
                        )

            t2 = default_timer()
            scheduler.step()
            
            # Log only on Rank 0
            if rank == 0:
                msg = f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
                logger.info(msg)

    else:
        # (Standard training loop logic mirrors AR loop above regarding DistributedSampler and logging)
        for ep in range(start_epoch, epochs):
            train_sampler.set_epoch(ep) # Crucial
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0
                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)
                
                pred = yy_tensor[..., :initial_step, :]
                inp_shape = list(xx_tensor.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, t_train):
                    inp = yy_tensor[..., t - initial_step : t, :].reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                    inp = inp.permute(temp_shape)
                    inp = torch.normal(inp, 0.001)
                    y = yy_tensor[..., t : t + 1, :]
                    temp_shape = [0]
                    temp_shape.extend(list(range(2, len(inp.shape))))
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    pred = torch.cat((pred, im), -2)

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy_tensor[..., :t_train, :]
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % model_update == 0 or ep == epochs:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)
                        pred = yy_tensor[..., :initial_step, :]
                        inp_shape = list(xx_tensor.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)
                        for t in range(initial_step, t_train):
                            inp = yy_tensor[..., t - initial_step : t, :].reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)
                            y = yy_tensor[..., t : t + 1, :]
                            temp_shape = [0]
                            temp_shape.extend(list(range(2, len(inp.shape))))
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2)
                            loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                            pred = torch.cat((pred, im), -2)
                        
                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy_tensor[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                    if dist.is_initialized():
                        val_loss_tensor = torch.tensor([val_l2_full], device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_l2_full = val_loss_tensor.item()

                    if rank == 0 and val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        torch.save(
                            {
                                "epoch": ep,
                                "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_val_min,
                            },
                            model_path,
                        )

            t2 = default_timer()
            scheduler.step()
            if rank == 0:
                msg = f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
                logger.info(msg)
    
    cleanup_distributed()

if __name__ == "__main__":
    run_training()
    if not dist.is_initialized() or dist.get_rank() == 0:
        msg = "Done."
        logger.info(msg)