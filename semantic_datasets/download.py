from registry import DownloaderRegistry

registry = DownloaderRegistry(data_dir="./data")

registry.register_defaults()  # registers OpenML + Kaggle

registry.download("openml/1464")
