
import random


def sample_frames(total_frames, frame_interval=10, frames_per_sample=10, frame_start_interval=1, obs_len=4, ensure_no_overlap=False):
    # 2hz 采样 => frame_interval=10
    # 4帧训练 预测6步 => frames_per_sample=10
    # 总帧数
    # if total_frames != 600:
    #     raise ValueError("图片列表必须包含600张图片(0000.jpg到0599.jpg)")

    # 每组采样10帧，所以每组需要的总帧跨度是9*10=90帧(因为第一帧到第十帧之间有9个间隔)
    # 所以最大起始帧是600-90=510
    max_start_frame = total_frames - (frame_interval * (frames_per_sample - 1)) - 1
    
    samples = []
    used_prefixes = set() if ensure_no_overlap else None

    for start in range(0, min(max_start_frame + 1, total_frames), frame_start_interval):
        # 生成10帧的序列 (间隔10帧)
        sample = [start + i * frame_interval for i in range(frames_per_sample)]
        
        is_overlap = False
        if ensure_no_overlap:
            for i_frame in sample[:obs_len]:
                if used_prefixes is not None and i_frame in used_prefixes:
                    is_overlap = True
                    break
            if not is_overlap:
                used_prefixes |= set(sample[:obs_len])
        if not is_overlap:
            samples.append(sample)
    
    return samples


def stratified_sampling(data, bins, ratios, sample_size, sample_key='trajectory_bias'):
    """
    按照value的分层比例进行采样
    
    Args:
        data: 原始数据列表，每个元素是一个包含'value'键的字典
        bins: 分桶边界列表，如[0, 0.1, 0.6, 1.0]
        ratios: 每个区间的采样比例，如[1, 3, 6]
        sample_size: 总采样数量
    
    Returns:
        采样后的子列表
    """
    # 1. 将数据分到不同的桶中
    buckets = [[] for _ in range(len(bins)-1)]
    
    for item in data:
        value = item[sample_key]
        print([value])
        print(bins)
        # 确定value属于哪个区间
        for i in range(len(bins)-1):
            if bins[i] <= value < bins[i+1]:
                buckets[i].append(item)
                break
        else:
            # 处理value等于最后一个边界的情况
            if value == bins[-1]:
                buckets[-1].append(item)
    
    # 2. 计算每个桶需要采样的数量
    if all([i > 0 for i in ratios]):
        total_ratio = sum(ratios)
        sample_counts = [int(sample_size * ratio / total_ratio) for ratio in ratios]

        # 处理由于取整可能导致的总数不足的情况
        remaining = sample_size - sum(sample_counts)
        for i in range(remaining):
            sample_counts[i % len(sample_counts)] += 1
    else:
        sample_counts = [int(i) for i in ratios]

    print('Buckets: ', [len(i_b) for i_b in buckets])
    print('sample_counts: ', sample_counts)
    # 3. 从每个桶中随机采样指定数量的样本
    sampled_data = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue  # 如果桶是空的，跳过
        # 如果需要采样的数量大于桶中的数量，则取全部
        sample_count = min(sample_counts[i], len(bucket)) if sample_counts[i] > 0 else len(bucket)
        sampled_data.extend(random.sample(bucket, sample_count))
    
    # 4. 打乱顺序以避免按桶排列
    random.shuffle(sampled_data)
    
    return sampled_data

