import argparse
import logging
import os
import pprint
import warnings

import torch
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import build_network
from tqdm import tqdm
from utils.criterion_helper import build_criterion
from utils.eval_helper import (
    Counting_TEMP,
    Localization_TEMP,
    dump,
    merge_together,
    performances,
)
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


def main():
    global \
        args, \
        config, \
        best_mae, \
        best_rmse, \
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
    config.temp_path = os.path.join(config.exp_path, "temp")
    if config.get("visualizer", None):
        config.visualizer.vis_dir = os.path.join(
            config.exp_path, config.visualizer.vis_dir
        )
        ifvis = config.visualizer.ifvis
        visualizer = build_visualizer(**config.visualizer)
    else:
        ifvis = False

    config.port = config.get("port", None)

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.temp_path, exist_ok=True)
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

    # parameters
    model.train()
    lr_scale_backbone = config.trainer["lr_scale_backbone"]
    if lr_scale_backbone == 0:
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad = False
        # parameters not include backbone
        parameters = [p for n, p in model.named_parameters() if "backbone" not in n]
    else:
        assert lr_scale_backbone > 0 and lr_scale_backbone <= 1
        parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": config.trainer.optimizer.kwargs.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_scale_backbone * config.trainer.optimizer.kwargs.lr,
            },
        ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    last_epoch = 0
    best_mae = 0
    best_rmse = 0
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
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]

    train_loader, val_loader, test_loader = build_dataloader(config.dataset)

    if args.evaluate:
        val_mae, val_mse, val_f1m_s, val_f1m_l = eval(
            val_loader,
            model,
            criterion,
            "val",
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
        train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch)
        lr_scheduler.step(epoch + 1)

        # validation
        if epoch > 20 and epoch % 3 == 0:
            val_mae, val_mse, val_f1m_s, val_f1m_l = eval(
                val_loader,
                model,
                criterion,
                "val",
                floc_path,
                gt_location_file,
                ifvis,
            )

            if best_f1m_l < val_f1m_l:
                logger.info("Model Saved!")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_metric": best_f1m_l,
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(config.save_path, "{}.pth".format(current_time)),
                )
                best_mae = val_mae
                best_rmse = val_mse
                best_f1m_l = val_f1m_l
                best_f1m_s = val_f1m_s


def train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch):
    model.train()
    if lr_scale_backbone == 0:
        model.backbone.eval()
        for p in model.backbone.parameters():
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
    train_loss = train_loss.item() / iter.item()


def eval(val_loader, model, criterion, type, floc_path, gt_location_file, ifvis):
    model.eval()
    logger = logging.getLogger("global_logger")

    logger.info("-----------------------------------------------------------")
    logger.info("Evaluation on val dataset or test dataset")

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

            dump(config.temp_path, outputs)

            density_pred = outputs["density_pred"]

            filename = outputs["filename"][0].split(".")[0]
            kpoint = Counting_TEMP(density_pred, filename, floc)

            if config.get("visualizer", None) and ifvis:
                visualizer.vis_batch(outputs, kpoint, filename)

    floc.close()
    floc_new = floc_path.replace(".txt", "_new.txt")
    gt_location_file = gt_location_file.replace("type", type)
    ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l = Localization_TEMP(
        floc_path, floc_new, gt_location_file
    )

    val_mae = None
    val_rmse = None
    gt_cnts, pred_cnts = merge_together(config.temp_path)
    val_mae, val_rmse = performances(gt_cnts, pred_cnts)

    # clean up temp files
    for file in os.listdir(config.temp_path):
        file_path = os.path.join(config.temp_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    logger.info("gather final results")

    logger.info(
        "Localization performance | AP_small: {} | AR_small: {} | F1m_small: {} | AP_large: {} | AR_large: {} | F1m_large: {}".format(
            ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l
        )
    )
    logger.info("Counting performance | MAE: {} | RMSE: {}".format(val_mae, val_rmse))
    logger.info(
        "Best Results | Best f1m_s: {}, Best f1m_l: {} | Best Val MAE: {}, Best Val RMSE: {}".format(
            best_f1m_s, best_f1m_l, best_mae, best_rmse
        )
    )
    logger.info("-----------------------------------------------------------")

    model.train()
    if lr_scale_backbone == 0:
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad = False

    return val_mae, val_rmse, f1m_s, f1m_l


if __name__ == "__main__":
    main()
