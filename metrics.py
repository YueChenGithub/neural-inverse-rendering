from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pathlib import Path
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import os
from tools.convert_img_channel import rgba2rgb

PSNR = PeakSignalNoiseRatio()
SSIM = StructuralSimilarityIndexMeasure()
LPIPS = LearnedPerceptualImagePatchSimilarity()


def eval_metrics(img1, img2):
    return PSNR(img1, img2), SSIM(img1, img2), LPIPS(img1, img2)


def metrics_test(log_dir, evaluate_dataset='test'):
    log_dir = Path(log_dir, 'test')
    subfolders = sorted(glob.glob(log_dir.__str__() + '/*/'))
    num_subfolders = len(subfolders)

    metrics_rerendering = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    metrics_albedo = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    for subfolder in subfolders:
        if evaluate_dataset in os.path.basename(os.path.dirname(subfolder)):
            # eval rerendering
            print(f'evaluating {os.path.basename(os.path.dirname(subfolder))}')
            pred = ToTensor()(Image.open(Path(subfolder, 'rgb.png')))[None, ...]
            gt = ToTensor()(Image.open(Path(subfolder, 'rgb_gt.png')))[None, ...]
            psnr, ssim, lpips = eval_metrics(pred, gt)
            metrics_rerendering['psnr'] += psnr / num_subfolders
            metrics_rerendering['ssim'] += ssim / num_subfolders
            metrics_rerendering['lpips'] += lpips / num_subfolders

            # eval albedo
            pred = ToTensor()(Image.open(Path(subfolder, 'Rd.png')))[None, ...]
            gt = ToTensor()(Image.open(Path(subfolder, 'Rd_gt.png')))[None, ...]
            psnr, ssim, lpips = eval_metrics(pred, gt)
            metrics_albedo['psnr'] += psnr / num_subfolders
            metrics_albedo['ssim'] += ssim / num_subfolders
            metrics_albedo['lpips'] += lpips / num_subfolders

    # eval env map
    pred = ToTensor()(Image.open(Path(log_dir, 'env_map.png')))[None, ...]
    gt = ToTensor()(Image.open(Path(log_dir, 'env_map_gt.png')))[None, ...]
    psnr_env = PSNR(pred, gt)

    # write metrics to txt
    with open(Path(log_dir, 'metrics_test.txt'), 'w') as f:
        # convert tensor into float
        metrics_rerendering = {k: v.item() for k, v in metrics_rerendering.items()}
        metrics_albedo = {k: v.item() for k, v in metrics_albedo.items()}
        # write to txt
        f.write(f'Number of evaluation: {len(subfolders)} {evaluate_dataset}\n')
        f.write('rerendering\n')
        f.write(str(metrics_rerendering))
        f.write('\n')
        f.write('albedo\n')
        f.write(str(metrics_albedo))
        f.write('\n')
        f.write('env\n')
        f.write(str(psnr_env.item()))

    # print result
    with open(Path(log_dir, 'metrics_test.txt')) as f:
        print(f.read())

def metrics_relighting(log_dir, data_dir, env_dir, evaluate_dataset='test'):
    log_dir = Path(log_dir, 'test')
    subfolders = sorted(glob.glob(log_dir.__str__() + '/*/'))
    num_subfolders = len(subfolders)
    env_list = glob.glob(str(Path(env_dir, "*.hdr")))
    env_name = []
    metrics_env = {}
    for env in env_list:
        env_name.append(os.path.basename(env)[:-4])
        metrics_env[os.path.basename(env)[:-4]] = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    print(metrics_env)

    for subfolder in subfolders:
        subfolder_name = os.path.basename(os.path.dirname(subfolder))
        if evaluate_dataset in subfolder_name:
            # eval rerendering
            print(f'evaluating {subfolder_name}')
            # # eval relihgting
            for env in env_name:
                print(f'evaluating {env}')
                pred = ToTensor()(Image.open(Path(subfolder, f'{env}.png')))[None, ...]
                gt = (Image.open(Path(data_dir, subfolder_name, f'rgba_{env}.png')))
                gt = rgba2rgb(gt, background=1)[None, ...]
                psnr, ssim, lpips = eval_metrics(pred, gt)
                metrics_env[env]['psnr'] += psnr / num_subfolders
                metrics_env[env]['ssim'] += ssim / num_subfolders
                metrics_env[env]['lpips'] += lpips / num_subfolders

    # write metrics to txt
    with open(Path(log_dir, 'metrics_relighting.txt'), 'w') as f:
        # convert tensor into float
        metrics_env = {k: {k_: v_.item() for k_, v_ in v.items()} for k, v in metrics_env.items()}
        # write to txt
        f.write(f'Number of evaluation: {len(subfolders)} {evaluate_dataset}\n')
        f.write('relighting\n')
        f.write(str(metrics_env))
        # write the mean psnr, ssim, lpips
        f.write('\n')
        f.write('mean\n')
        mean_psnr = 0
        mean_ssim = 0
        mean_lpips = 0
        for env in env_name:
            mean_psnr += metrics_env[env]['psnr'] / len(env_name)
            mean_ssim += metrics_env[env]['ssim'] / len(env_name)
            mean_lpips += metrics_env[env]['lpips'] / len(env_name)
        f.write(f'psnr: {mean_psnr}\n')
        f.write(f'ssim: {mean_ssim}\n')
        f.write(f'lpips: {mean_lpips}\n')


    # print result
    with open(Path(log_dir, 'metrics_relighting.txt')) as f:
        print(f.read())

if __name__ == '__main__':
    log_dir = '/home/yue/Desktop/ma2/log/test_run/hotdog'
    env_dir = '/home/yue/Desktop/ma2/light_probe/relighting_env_map/'
    data_dir = '/home/yue/Desktop/ma2/dataset/nerfactor/hotdog_2163'
    evaluate_dataset = 'val'

    metrics_test(log_dir, evaluate_dataset)
    metrics_relighting(log_dir, data_dir, env_dir, evaluate_dataset)
















