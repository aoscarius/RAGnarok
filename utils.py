import os
import requests
from tqdm import tqdm

def modelDownload(url: str, dest_folder: str, dest_name: str) -> str:
    dest_path = os.path.join(dest_folder, dest_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
        if not os.path.exists(dest_path):
            print(f"Model {dest_name} not found.")
            response = requests.get(url, stream=True)
            total = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as file, tqdm(
                desc=f"Downloading {dest_name}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
   
    return dest_path

