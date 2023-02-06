import torch


def mpjpe_error(batch_imp, batch_gt):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))


def mpjpe_error_bh(batch_imp, batch_gt, eval_points):
    total_error = 0
    for i in range(len(batch_imp)):
        seq = batch_imp[i]
        seq_error = 0
        num_frames = 0
        for j in range(len(seq)):
            frame = seq[j]
            frame_error = 0
            for k in range(0, len(frame), 3):
                x_imp = batch_imp[i, j, k]
                y_imp = batch_imp[i, j, k + 1]
                z_imp = batch_imp[i, j, k + 2]
                x_gt = batch_gt[i, j, k]
                y_gt = batch_gt[i, j, k + 1]
                z_gt = batch_gt[i, j, k + 2]

                error = (x_imp - x_gt) ** 2 + (y_imp - y_gt) ** 2 + (z_imp - z_gt) ** 2
                error = np.sqrt(error)

                frame_error += error
            missing_joints = eval_points[i, j].sum().item() / 3
            if missing_joints > 0:
                frame_error /= missing_joints
                num_frames += 1
            else:
                frame_error = 0
            seq_error += frame_error
        seq_error /= num_frames
        total_error += seq_error
    total_error /= len(batch_imp)

    return total_error


def mpjpe_error_l2(batch_imp, batch_gt):
    total_error = 0
    for i in range(len(batch_imp)):
        seq_imp = batch_imp[i]
        seq_gt = batch_gt[i]
        seq_error = torch.norm(seq_imp - seq_gt).item()
        total_error += seq_error
    total_error /= len(batch_imp)

    return total_error
