import os
import json
from pathlib import Path
import requests
from tqdm import tqdm
from typing import List, Dict, Optional

class ModelDownloader:
    def __init__(self, model_data_path: str = "models/model_data.json", weights_dir: str = "pretrained_checkpoint"):
        self.model_data_path = Path(model_data_path)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)

    def load_model_data(self) -> List[Dict]:
        """Load model information from JSON file."""
        if not self.model_data_path.exists():
            raise FileNotFoundError(f"Model data file not found at {self.model_data_path}")
        
        with open(self.model_data_path, 'r') as f:
            return json.load(f)

    def download_file(self, url: str, destination: Path) -> None:
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

    def get_model_path(self, url: str) -> Path:
        """Get the path where a model should be saved based on the download URL."""
        filename = Path(url).name
        return self.weights_dir / filename

    def download_model(self, model_info: Dict) -> Path:
        """Download a specific model."""
        url = model_info["weights_link"]
        model_path = self.get_model_path(url)
        
        if not model_path.exists():
            print(f"Downloading {model_info['Model']} ({model_info['Size']})...")
            print(f"Parameters: {model_info['Number of parameters']}, AP: {model_info['AP']}, FPS: {model_info['FPS']}")
            self.download_file(url, model_path)
        else:
            print(f"Model {model_info['Model']} already exists, skipping...")
        
        return model_path

    def download_all_models(self) -> List[Path]:
        """Download all available models."""
        model_data = self.load_model_data()
        downloaded_paths = []
        
        for model_info in model_data:
            model_path = self.download_model(model_info)
            downloaded_paths.append(model_path)
        
        return downloaded_paths

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        model_data = self.load_model_data()
        return next((model for model in model_data if model["Model"] == model_name), None)

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        model_info = self.get_model_info(model_name)
        if model_info:
            url = model_info["weights_link"]
            model_path = self.get_model_path(url)
            return model_path.exists()
        return False

if __name__ == "__main__":
    downloader = ModelDownloader()
    downloader.download_all_models()
