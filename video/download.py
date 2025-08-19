import os
import time
import random
import requests

file_path = './vid.txt'
download_dir = '.'  # 指定下载目录

# 确保下载目录存在
os.makedirs(download_dir, exist_ok=True)

def download_video(url):
    try:
        file_name = os.path.join(download_dir, url.split('/')[-1])
        response = requests.get(url, stream=True,verify=False)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"下载成功: {url}")
        else:
            print(f"下载失败: {url}, HTTP 状态码: {response.status_code}")
    except Exception as e:
        print(f"下载失败: {url}, 错误: {e}")

# 读取 txt 文件中的所有链接
with open(file_path, 'r', encoding='utf-8') as f:
    video_links = [line.strip() for line in f.readlines() if line.strip()]

# 遍历指定范围的链接进行下载
for i in range(len(video_links)):
    print(video_links[i])
    download_video(video_links[i])
    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)
