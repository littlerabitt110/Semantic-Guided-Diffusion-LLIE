import os
import math
import argparse
import random
import logging
import time
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import options.options as option
from utils import util
import torchvision.transforms as T
import model as Model
import core.logger as Logger
import core.metrics as Metrics
import natsort
from torchvision import transforms

# 归一化处理
transform = transforms.Lambda(lambda t: (t * 2) - 1)


def generate_images(diffusion, image_paths, device, result_path):
    """
    处理一批图片，并统计推理时间
    """
    total_time = 0
    num_images = len(image_paths)

    for i, image_path in enumerate(image_paths):
        raw_img = Image.open(image_path).convert('RGB')
        img_w, img_h = raw_img.size
        raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)
        raw_img = transform(TF.to_tensor(raw_img)).unsqueeze(0).to(device)

        val_data = {'LQ': raw_img, 'GT': raw_img}
        diffusion.feed_data(val_data)

        # 计时开始
        start_time = time.time()
        diffusion.test(continous=False)
        end_time = time.time()

        total_time += (end_time - start_time)

        visuals = diffusion.get_current_visuals()
        normal_img = Metrics.tensor2img(visuals['HQ'])
        normal_img = cv2.resize(normal_img, (img_w, img_h))

        # 生成文件名
        filename, ext = os.path.splitext(os.path.basename(image_path))
        new_filename = f"{filename}_normal{ext}"

        # 保存生成图片
        util.save_img(normal_img, os.path.join(result_path, new_filename))

        print(f"[{i + 1}/{num_images}] 处理完成: {new_filename}")

    avg_time = total_time / num_images
    print(f"总推理时间: {total_time:.2f} 秒, 单张图片平均时间: {avg_time:.4f} 秒")


def main():
    """
    运行主测试脚本
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./config/dataset.yml', help='Path to dataset config file')
    parser.add_argument('--input', type=str, default='images/unpaired/', help='Path to input images')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to test')
    parser.add_argument('--gpu_ids', type=str, default="0", help='GPU IDs')
    parser.add_argument('-c', '--config', type=str, default='config/test_unpaired.json', help='JSON config file')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], default='train', help='Phase: train/val')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    opt['phase'] = 'test'
    opt['dist'] = False

    util.setup_logger('base', opt['path']['log'], 'test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    # 设定随机种子
    seed = opt['train']['manual_seed'] or random.randint(1, 10000)
    util.set_random_seed(seed)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Model.create_model(opt).to(device)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    logger.info('Model loaded successfully')

    # 结果保存路径
    result_path = opt['path']['results']
    os.makedirs(result_path, exist_ok=True)

    # 读取输入图片
    input_path = args.input
    image_files = natsort.natsorted(
        [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('png', 'jpg', 'jpeg'))])

    if args.num_images > len(image_files):
        print(f"警告: 目录中只有 {len(image_files)} 张图片，无法测试 {args.num_images} 张")
        args.num_images = len(image_files)

    print(f"开始测试 {args.num_images} 张图片...")
    generate_images(diffusion, image_files[:args.num_images], device, result_path)


if __name__ == '__main__':
    main()
