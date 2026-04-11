import os

def check_dataset_pairing(root_dir):
    low_dir = os.path.join(root_dir, "low")
    high_dir = os.path.join(root_dir, "high")
    seg_dir = os.path.join(root_dir, "segmentation")

    low_files = sorted(os.listdir(low_dir))
    high_files = sorted(os.listdir(high_dir))
    seg_files = sorted(os.listdir(seg_dir))

    #print("Low 文件夹数量:", len(low_files))
    #print("High 文件夹数量:", len(high_files))
    #print("Segmentation 文件夹数量:", len(seg_files))
    print("Files in High==Low==Segmentation")

    # 检查数量是否一致并正好为490组
    # if len(low_files) == len(high_files) == len(seg_files) == 490:
    if len(low_files) == len(high_files)  == 1490:
        print("----Paired Dataset Matched----")
    else:
        print("!!!!--Paired Dataset Unmatched--!!!")

    # 检查配对情况
    mismatches = []
    for f in low_files:
        base, ext = os.path.splitext(f)
        expected_high = f  # 期望High文件夹中文件名与Low一致
        expected_seg = base + "_cropped_vis" + ext  # 期望Segmentation中文件名为 Low 文件名加 _mask 后缀
        if expected_high not in high_files:
            mismatches.append(f"High文件夹缺失文件: {expected_high}")
        if expected_seg not in seg_files:
            mismatches.append(f"Segmentation文件夹缺失文件: {expected_seg}")

    if mismatches:
        print("发现配对问题:")
        for m in mismatches:
            print(m)
    else:
        print("所有文件均正确配对。")

# 设置数据集根目录（根据实际情况修改路径）
root_dir = "dataset/LOLv1/our485"
check_dataset_pairing(root_dir)
