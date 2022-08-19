import os
import numpy as np
from tqdm import tqdm
import requests
from typing import List, Tuple

base_dir = os.path.dirname(__file__)


def download_url(url: str, folder_name: str):
    buffer_size = 1024  # read 1024 bytes every time
    response = requests.get(url, stream=True)   # download the body of response by chunk, not immediately
    file_size = int(response.headers.get("Content-Length", 0))  # get total file size
    filename = url.split("/")[-1]   # get file name

    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(buffer_size), f"Downloading {filename}", total=file_size, unit="B",
                    unit_scale=True, unit_divisor=1024)
    path = f'{base_dir}/Data/{folder_name}/'
    if os.path.exists(path) is False:
        os.makedirs(path)
    with open(path + filename, "wb") as f:
        for data in progress:
            f.write(data)               # write data read to the file
            progress.update(len(data))  # update the progress bar manually


def download_data(urls: List[str], folder_name: str):
    assert os.path.exists(f'{base_dir}/Data'), 'wrong base directory, accessed externally'
    if os.path.exists(f'{base_dir}/Data/{folder_name}/') is False:
        print('Download data\n')
        for url in urls:
            download_url(url, folder_name)
    else:
        print('Data already loaded.')


def jackknife(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(arr, np.ndarray) is False:
        arr = np.array(arr.data)
    assert arr.ndim == 1
    subs = []
    for i in range(len(arr)):
        subsample = np.delete(arr, i)
        subs.append(np.mean(subsample))
    subs = np.asarray(subs)
    mean = np.mean(subs)
    err = np.sqrt((len(arr)-1) * np.mean(np.square(subs - mean)))
    return mean, err
