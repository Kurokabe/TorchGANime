from datasets import load_dataset, Image
dataset = load_dataset("imagefolder", data_dir="/TorchGANime/data/kny/images/01/", drop_labels=True).cast_column("image", Image())
dataset.push_to_hub("Kurokabe/Kimetsu-no-Yaiba-Image-Dataset-01")