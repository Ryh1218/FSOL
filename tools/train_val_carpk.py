import argparse
import logging
import os
import pprint
import warnings

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.data_builder import build_dataloader
from models.model_helper import build_network
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import Counting, Localization
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    create_logger,
    get_current_time,
    set_random_seed,
    to_device,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import build_visualizer

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="FSOL train and evaluation")
parser.add_argument(
    "-c", "--config", type=str, default="./config.yaml", help="Path of config"
)
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")


def main():
    global \
        args, \
        config, \
        best_mae, \
        best_mse, \
        best_f1m_l, \
        best_f1m_s, \
        visualizer, \
        lr_scale_backbone

    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    if config.get("visualizer", None):
        config.visualizer.vis_dir = os.path.join(
            config.exp_path, config.visualizer.vis_dir
        )
        ifvis = config.visualizer.ifvis
        visualizer = build_visualizer(**config.visualizer)
    else:
        ifvis = False

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        if (args.evaluate or args.test) and config.get("visualizer", None):
            os.makedirs(config.visualizer.vis_dir, exist_ok=True)
            
        current_time = get_current_time()
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("\nargs: {}".format(pprint.pformat(args)))
        logger.info("\nconfig: {}".format(pprint.pformat(config)))

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    criterion = build_criterion(config.criterion)

    # create model
    model = build_network(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # parameters
    model.train()
    lr_scale_backbone = config.trainer["lr_scale_backbone"]
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False
        # parameters not include backbone
        parameters = [
            p for n, p in model.module.named_parameters() if "backbone" not in n
        ]
    else:
        assert lr_scale_backbone > 0 and lr_scale_backbone <= 1
        parameters = [
            {
                "params": [
                    p
                    for n, p in model.module.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": config.trainer.optimizer.kwargs.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.module.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_scale_backbone * config.trainer.optimizer.kwargs.lr,
            },
        ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    last_epoch = 0
    best_mae = 0
    best_mse = 0
    best_f1m_s = 0
    best_f1m_l = 0

    gt_files_folder = config.files.get("gt_files_folder", None)
    gt_location_file = os.path.join(gt_files_folder, "sf_type_gt.txt")
    floc_path = os.path.join(gt_files_folder, "localization_type.txt")

    load_weight = config.saver.get("load_weight", None)
    ifload = config.saver.get("ifload", False)

    if ifload and load_weight:
        logger.info(
            "=> loading checkpoint '{}'".format(
                os.path.join(config.save_path, load_weight)
            )
        )
        checkpoint = torch.load(os.path.join(config.save_path, load_weight))
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]

    train_loader, val_loader, test_loader = build_dataloader(
        config.dataset, distributed=True
    )

    if args.evaluate:
        val_mae, val_mse, val_f1m_s, val_f1m_l = eval(
            val_loader,
            model,
            criterion,
            "test",
            floc_path,
            gt_location_file,
            ifvis,
        )
        return

    if args.test:
        test_mae, test_mse, test_f1m_s, test_f1m_l = eval(
            test_loader,
            model,
            criterion,
            "test",
            floc_path,
            gt_location_file,
            ifvis,
        )
        return

    for epoch in range(last_epoch, config.trainer.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch)
        lr_scheduler.step(epoch + 1)

        # validation
        if epoch % 3 == 0 or epoch + 1 == config.trainer.epochs:
            val_mae, val_mse, val_f1m_s, val_f1m_l = eval(
                val_loader,
                model,
                criterion,
                "test",
                floc_path,
                gt_location_file,
                ifvis,
            )

            if rank == 0:
                if best_f1m_l < val_f1m_l:
                    logger.info("Model Saved!")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_metric": best_f1m_l,
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(config.save_path, "ckpt.pth.tar"),
                    )
                    best_mae = val_mae
                    best_mse = val_mse
                    best_f1m_l = val_f1m_l
                    best_f1m_s = val_f1m_s


def train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch):
    model.train()
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False

    logger = logging.getLogger("global_logger")
    logger.info(
        "Start Train Epoch : {} / {}".format(
            epoch + 1,
            config.trainer.epochs,
        )
    )
    train_loss = 0

    for i, sample in enumerate(tqdm(train_loader)):
        iter = i + 1
        sample = to_device(sample, device=torch.device("cuda"))
        # forward
        outputs = model(sample)  # 1 x 1 x h x w
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = torch.Tensor([train_loss]).cuda()
    iter = torch.Tensor([iter]).cuda()
    dist.all_reduce(train_loss)
    dist.all_reduce(iter)
    train_loss = train_loss.item() / iter.item()


def eval(
    val_loader, model, criterion, type, floc_path, gt_location_file, ifvis
):
    model.eval()
    logger = logging.getLogger("global_logger")
    rank = dist.get_rank()
    if rank == 0:
        logger.info("-----------------------------------------------------------")
        logger.info("Evaluation on val dataset or test dataset")

    dist.barrier()

    floc_path = floc_path.replace("type", type)
    floc = open(floc_path, "w+")

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_loader)):
            sample = to_device(sample, device=torch.device("cuda"))
            outputs = model(sample)
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)

            filename = outputs["filename"][0].split(".")[0]
            filename = filename.split('_')[1] + '_' + filename.split('_')[2]
            kpoint = Counting(outputs["density_pred"], filename, floc, False)

            if config.get("visualizer", None) and ifvis:
                visualizer.vis_batch(outputs, kpoint, filename)

    floc.close()
    floc_new = floc_path.replace(".txt", "_new.txt")
    gt_location_file = gt_location_file.replace("type", type)
    ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l, mae, mse = Localization(
        floc_path, floc_new, gt_location_file, True
    )

    dist.barrier()
    if rank == 0:
        logger.info("gather final results")

    if rank == 0:
        logger.info(
            "Localization performance | AP_small: {} | AR_small: {} | F1m_small: {} | AP_large: {} | AR_large: {} | F1m_large: {}".format(
                ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l
            )
        )
        logger.info(
            "Counting performance | MAE: {} | RMSE: {}".format(
                mae, mse
            )
        )
        logger.info(
            "Finish Val | f1m_s: {}, f1m_l: {} | Best f1m_s: {}, Best f1m_l: {} | Val MAE: {}, Val MSE: {} | Best Val MAE: {}, Best Val MSE: {}".format(
                f1m_s, f1m_l, best_f1m_s, best_f1m_l, mae, mse, best_mae, best_mse
            )
        )
        logger.info("-----------------------------------------------------------")

    model.train()
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False

    return mae, mse, f1m_s, f1m_l


if __name__ == "__main__":
    main()
