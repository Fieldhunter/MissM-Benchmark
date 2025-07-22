import os
from pathlib import Path


def mp4_to_wav(mp4_path, wav_path, sampling_rate):
    if os.path.exists(wav_path):
        os.remove(wav_path)
    Path("/".join(wav_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)

    command = "ffmpeg -i {} -ac 1 -ar {} {}".format(mp4_path.replace(' ', '\ '), sampling_rate, wav_path.replace(' ', '\ '))
    print('执行命令: ', command)
    os.system(command)


if __name__ == '__main__':
    mp4_dir = '/big-data/person/yuanjiang/MLMM_datasets/AVE_Dataset/ave_train_split'
    wav_dir = '/big-data/person/yuanjiang/MLMM_datasets/AVE_Dataset/ave_train_split_wav'
    sampling_rate = 16000

    for file_path in Path(mp4_dir).rglob("*"):  # `*` 匹配所有文件和子目录
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts) and file_path.suffix.lower() in ['.mp4', '.avi']:
            mp4_to_wav(str(file_path), str(file_path).replace(mp4_dir, wav_dir).replace(file_path.suffix.lower(), '.wav'), sampling_rate)
