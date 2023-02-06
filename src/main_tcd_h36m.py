from typing import *
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.model_h36m import ModelH36M
from src.data.h36m import H36M
from utils.parser import args


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    valid_epoch_interval: int = 5,
    foldername: str = ".",
) -> None:
    """Train the model.
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch._utils.DataLoader): Training data loader.
        valid_loader (torch._utils.DataLoader, optional): Validation data loader. Defaults to None.
        valid_epoch_interval (int, optional): Interval for validation. Defaults to 5.
        foldername (str, optional): Folder name to save the model. Defaults to "".
    """
    model.train()
    output_path = foldername + "/model.pth"

    # Optimizer.
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ## Scheduler.
    p1 = int(0.75 * args.epochs)
    p2 = int(0.9 * args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    # Train.
    train_loss = []
    valid_loss = []
    best_valid_loss = 1e10

    # Iterate over epochs.
    for epoch in range(args.epochs):
        avg_loss = 0

        with tqdm(
            train_loader, mininterval=5.0, maxinterval=50.0, desc=f"Epoch {epoch}"
        ) as pbar_train:
            for i, batch in enumerate(pbar_train, start=1):
                optimizer.zero_grad()

                # TODO the inference already computes the loss :/
                loss = model(batch, train=True).mean()
                loss.backward()
                avg_loss += loss.item()

                optimizer.step()

                pbar_train.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / i,
                        "epoch": epoch,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        train_loss.append(
            avg_loss / len(pbar_train)
        )  # TODO: the length of pbar, batch_no and pbar.n should all be the same.
        if valid_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as pbar_val:
                    for batch_no, valid_batch in enumerate(pbar_val, start=1):
                        batch = valid_batch
                        loss = model(batch, is_train=0).mean()
                        avg_loss_valid += loss.item()
                        pbar_val.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch,
                            },
                            refresh=False,
                        )

            valid_loss.append(avg_loss_valid / len(pbar_val))

            # Best loss.
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / len(pbar_val),
                    "at",
                    epoch,
                )
                torch.save(model.state_dict(), output_path)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(np.arange(1, len(train_loss) + 1), train_loss)
            ax.plot(np.arange(1, len(valid_loss) + 1) * 5, valid_loss)
            ax.grid(True)
            plt.show()
            fig.savefig(f"{foldername}/loss_{epoch}.png")

    torch.save(model.state_dict(), output_path)
    np.save(f"{foldername}/train_loss.npy", np.array(train_loss))
    np.save(f"{foldername}/valid_loss.npy", np.array(valid_loss))


def evaluate(model_s, model_l, loader, nsample=5, scaler=1, sample_strategy="best"):
    with torch.no_grad():
        model_s.eval()
        model_l.eval()
        mse_total = 0
        mae_total = 0
        mpjpe_total = 0
        mpjpe_bh_total = 0
        mpjpe_l2_total = 0
        evalpoints_total = 0

        mse_all = []
        mae_all = []
        mpjpe_all = []
        mpjpe_all_bh = []
        mpjpe_all_l2 = []

        all_target = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        dim_used = np.array(
            [
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                63,
                64,
                65,
                66,
                67,
                68,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                87,
                88,
                89,
                90,
                91,
                92,
            ]
        )

        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate(
            (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2)
        )
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate(
            (joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2)
        )

        titles = np.array(range(output_n)) + 1
        m_p3d_h36 = np.zeros([output_n])
        n = 0

        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                batch = test_batch
                batch_size = batch["pose"].shape[0]
                n += batch_size

                s = {
                    "pose": batch["pose"].clone()[:, : input_n + 5],
                    "mask": batch["mask"].clone()[:, : input_n + 5],
                    "timepoints": batch["timepoints"].clone()[:, : input_n + 5],
                }

                output = model_s.module.evaluate(s, nsample)
                samples, _, _, _ = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples_mean = np.mean(samples.cpu().numpy(), axis=1)
                batch["pose"][:, : input_n + 5] = torch.from_numpy(samples_mean)
                batch["mask"][:, : input_n + 5] = 1

                output = model_l.module.evaluate(batch, nsample)

                all_joints_seq = batch["pose_32"].clone()

                samples, c_target, eval_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = batch["pose_32"]  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)

                samples_mean = np.mean(samples.cpu().numpy(), axis=1)

                renorm_pose = []
                renorm_c_target = []
                renorm_all_joints_seq = []

                for i in range(len(samples_mean)):
                    renorm_all_joints_i = all_joints_seq.cpu().data.numpy()[i][
                        input_n : input_n + output_n
                    ]
                    renorm_c_target_i = c_target.cpu().data.numpy()[i][
                        input_n : input_n + output_n
                    ]

                    if sample_strategy == "best":
                        best_renorm_pose = None
                        best_error = float("inf")

                        for j in range(nsample):
                            renorm_pose_j = (
                                samples.cpu().numpy()[i][j][
                                    input_n : input_n + output_n
                                ]
                                * 1000
                            )
                            renorm_all_joints_j = renorm_all_joints_i.copy()
                            renorm_all_joints_j[:, dim_used] = renorm_pose_j
                            renorm_all_joints_j[
                                :, index_to_ignore
                            ] = renorm_all_joints_j[:, index_to_equal]
                            error = mpjpe_error(
                                torch.from_numpy(renorm_all_joints_j).view(
                                    output_n, 32, 3
                                ),
                                torch.from_numpy(renorm_c_target_i).view(
                                    output_n, 32, 3
                                ),
                            )
                            if error.item() < best_error:
                                best_error = error.item()
                                best_renorm_pose = renorm_pose_j
                    else:
                        best_renorm_pose = (
                            samples_mean[i][input_n : input_n + output_n] * 1000
                        )
                    renorm_pose.append(best_renorm_pose)
                    renorm_c_target.append(renorm_c_target_i)
                    renorm_all_joints_seq.append(renorm_all_joints_i)

                renorm_pose = torch.from_numpy(np.array(renorm_pose))
                renorm_c_target = torch.from_numpy(np.array(renorm_c_target))
                renorm_all_joints_seq = torch.from_numpy(
                    np.array(renorm_all_joints_seq)
                )

                renorm_all_joints_seq[:, :, dim_used] = renorm_pose
                renorm_all_joints_seq[:, :, index_to_ignore] = renorm_all_joints_seq[
                    :, :, index_to_equal
                ]

                mpjpe_p3d_h36 = torch.sum(
                    torch.mean(
                        torch.norm(
                            renorm_c_target.view(-1, output_n, 32, 3)
                            - renorm_all_joints_seq.view(-1, output_n, 32, 3),
                            dim=3,
                        ),
                        dim=2,
                    ),
                    dim=0,
                )
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

                eval_points = eval_points[:, input_n : input_n + output_n, :]

                all_target.append(renorm_c_target)
                all_evalpoint.append(eval_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(renorm_all_joints_seq)

                mse_current = (
                    (
                        (
                            renorm_pose.to(device)
                            - renorm_c_target[:, :, dim_used].to(device)
                        )
                        * eval_points
                    )
                    ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs(
                        (
                            renorm_pose.to(device)
                            - renorm_c_target[:, :, dim_used].to(device)
                        )
                        * eval_points
                    )
                ) * scaler

                mpjpe_current = mpjpe_error(
                    renorm_all_joints_seq.view(-1, output_n, 32, 3),
                    renorm_c_target.view(-1, output_n, 32, 3),
                )
                mpjpe_current_bh = mpjpe_error_bh(
                    renorm_pose, renorm_c_target[:, :, dim_used], eval_points
                )
                mpjpe_current_l2 = mpjpe_error_l2(
                    renorm_pose, renorm_c_target[:, :, dim_used]
                )

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mpjpe_total += mpjpe_current.item()
                mpjpe_bh_total += mpjpe_current_bh.item()
                mpjpe_l2_total += mpjpe_current_l2
                evalpoints_total += eval_points.sum().item()

                mse_all.append(
                    np.sqrt(mse_current.sum().item() / eval_points.sum().item())
                )
                mae_all.append(mae_current.sum().item() / eval_points.sum().item())
                mpjpe_all.append(mpjpe_current.item())
                mpjpe_all_bh.append(mpjpe_current_bh.item())
                mpjpe_all_l2.append(mpjpe_current_l2)

                it.set_postfix(
                    ordered_dict={
                        "average_rmse": np.sqrt(mse_total / evalpoints_total),
                        "average_mae": mae_total / evalpoints_total,
                        "average_mpjpe": mpjpe_total / batch_no,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            print("Average MPJPE:", mpjpe_total / batch_no)

            ret = {}
            m_p3d_h36 = m_p3d_h36 / n
            for j in range(output_n):
                ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

            return all_generated_samples, all_target, all_evalpoint, ret


def main():
    # * Train.
    if args.mode == "train":
        model = ModelH36M(
            config=config, device=device, target_dim=(args.joints * args.channels)
        )

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        # Data.
        all_data = True if args.data == "all" else False

        train_ds = H36M(
            data_dir=args.data_dir,
            input_n=args.input_n,
            output_n=args.output_n,
            skip_rate=args.skip_rate,
            split=0,
            miss_rate=(args.miss_rate / 100),
            all_data=all_data,
            joints=args.joints,
        )
        print(">>> Training dataset length: {:d}".format(train_ds.__len__()))
        train_dl = DataLoader(
            train_ds,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        valid_ds = H36M(
            data_dir=args.data_dir,
            input_n=args.input_n,
            output_n=args.output_n,
            skip_rate=args.skip_rate,
            split=1,
            miss_rate=(args.miss_rate / 100),
            all_data=all_data,
            joints=args.joints,
        )
        print(">>> Validation dataset length: {:d}".format(valid_ds.__len__()))
        valid_dl = DataLoader(
            valid_ds,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Loop.
        train(
            model=model,
            train_loader=train_dl,
            valid_loader=valid_dl,
            foldername=args.output_dir,
        )

    # * Test.
    elif args.mode == "test":
        actions = [
            "walking",
            "eating",
            "smoking",
            "discussion",
            "directions",
            "greeting",
            "phoning",
            "posing",
            "purchases",
            "sitting",
            "sittingdown",
            "takingphoto",
            "waiting",
            "walkingdog",
            "walkingtogether",
        ]

        model_s = Model_H36M(config, device, target_dim=(args.joints * 3))
        model_l = Model_H36M(config, device, target_dim=(args.joints * 3))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model_s = nn.DataParallel(model_s)
            model_l = nn.DataParallel(model_l)

        model_s.to(device)
        model_l.to(device)

        model_s.load_state_dict(torch.load(f"{args.model_s}/model_s.pth"))
        model_l.load_state_dict(torch.load(f"{args.model_l}/model_l.pth"))

        head = np.array(["act"])
        for k in range(1, output_n + 1):
            head = np.append(head, [f"#{k}"])
        errs = np.zeros([len(actions) + 1, output_n])

        for i, action in enumerate(actions):
            test_dataset = H36M(
                data_dir,
                input_n,
                output_n,
                skip_rate,
                split=2,
                miss_rate=(args.miss_rate / 100),
                joints=args.joints,
                actions=[action],
            )
            print(">>> Test dataset length: {:d}".format(test_dataset.__len__()))
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["train"]["batch_size_test"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            pose, target, mask, ret = evaluate(
                model_s,
                model_l,
                test_loader,
                nsample=5,
                scaler=1,
                sample_strategy="best",
            )

            ret_log = np.array([])
            for k in ret.keys():
                ret_log = np.append(ret_log, [ret[k]])
            errs[i] = ret_log

        errs[-1] = np.mean(errs[:-1], axis=0)
        actions = np.expand_dims(np.array(actions + ["average"]), axis=1)
        value = np.concatenate([actions, errs.astype(np.str)], axis=1)
        save_csv_log(head, value, is_create=True, file_name="fde_per_action")


if __name__ == "__main__":
    # TODO: Remove personal comments.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Config dictionary.
    config = {
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            # "batch_size_test": 16, # ? Is it necessary?
            "lr": args.lr,
        },
        "diffusion": {
            "layers": args.diffusion_layers,
            "channels": args.diffusion_channels,
            "nheads": args.diffusion_heads,
            "diffusion_embedding_dim": args.diffusion_embedding,
            "beta_start": args.diffusion_beta_start,
            "beta_end": args.diffusion_beta_end,
            "num_steps": args.diffusion_timesteps,
            "schedule": args.variance_scheduler,
        },
        "model": {
            "conditional": args.conditional,
            "timeemb": args.time_embedding,
            "featureemb": args.feature_embedding,
        },
    }

    main()
