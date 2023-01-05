from accelerate import Accelerator
from transformers import get_scheduler, TrainingArguments
from torchganime.models.ganime import GANime, GANimeConfig
from torchganime.data.dataloader.video import VideoData
from transformers import GPT2Config
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)


dataset = VideoData(
    "/TorchGANime/data/kny/raw_videos/02.mkv",
    "/TorchGANime/data/snk/raw_videos/02.mkv",
    image_size=(128, 256),
    batch_size=training_args.per_device_train_batch_size,
    num_workers=16,
)
accelerator = Accelerator()
model = GANime(
    GANimeConfig(
        "/TorchGANime/checkpoints/vqgan_full/checkpoints/epoch=199-step=168800.ckpt",
        GPT2Config(),
        transformer_ckpt_path="gpt2",
    )
)
optimizer = AdamW(model.parameters(), lr=3e-5)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    dataset.train_dataloader(), dataset.val_dataloader(), model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))

model.train()

for step, batch in enumerate(train_dataloader, start=1):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(batch)
    loss = outputs["loss"] / training_args.gradient_accumulation_steps
    accelerator.backward(loss)

    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    progress_bar.update(1)
