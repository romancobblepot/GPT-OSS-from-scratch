#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,SequentialLR,CosineAnnealingLR
import time
from tqdm import tqdm
import os


# In[15]:


best_model_params_path = "best-gpt-oss-params.pt"


# In[14]:


def calc_loss_total(
    data_loader,
    model,
    device="cuda",
    num_batches=None
):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(
            num_batches,
            len(data_loader)
        )

    model.eval()

    with torch.no_grad():

        for i, (input_batch, target_batch) in enumerate(data_loader):

            if i >= num_batches:
                break

            input_batch = input_batch.to(
                device,
                non_blocking=True
            )

            target_batch = target_batch.to(
                device,
                non_blocking=True
            )

            output_batch, _ = model(input_batch)

            loss = torch.nn.functional.cross_entropy(
                output_batch.flatten(0, 1),
                target_batch.flatten()
            )

            total_loss += loss.item()

    return total_loss / num_batches


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    eval_freq,
    eval_iter,
    gradient_accumulation_steps,
    device="cuda"
):

    global_step = 0

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        model.train()

        optimizer.zero_grad(set_to_none=True)

        for input_batch, target_batch in tqdm(train_loader):

            input_batch = input_batch.to(
                device,
                non_blocking=True
            )

            target_batch = target_batch.to(
                device,
                non_blocking=True
            )

            output_batch, _ = model(input_batch)

            loss = torch.nn.functional.cross_entropy(
                output_batch.flatten(0, 1),
                target_batch.flatten()
            )

            # Gradient accumulation
            loss = loss / gradient_accumulation_steps

            loss.backward()

            global_step += 1

            if global_step % gradient_accumulation_steps == 0:

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            # Evaluation
            if global_step % eval_freq == 0:

                model.eval()

                train_loss = calc_loss_total(
                    train_loader,
                    model,
                    device=device,
                    num_batches=eval_iter
                )

                val_loss = calc_loss_total(
                    val_loader,
                    model,
                    device=device,
                    num_batches=eval_iter
                )

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    torch.save(
                        model.state_dict(),
                        best_model_params_path
                    )

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Ep {epoch + 1} "
                    f"(step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f}, "
                    f"Val loss: {val_loss:.3f}"
                )

                model.train()

    return train_losses, val_losses


# In[13]:


def trainer(model, train_loader, val_loader):

    learning_rate = 3e-4

    num_epochs = 1

    warmup_steps = 100

    min_lr = 3e-5

    eval_freq = 150

    eval_iters = 5

    gradient_accumulation_steps = 10

    device = "cuda"

    total_steps = (
        num_epochs
        * len(train_loader)
        // gradient_accumulation_steps
    )

    # Load previous checkpoint if available
    if os.path.exists(best_model_params_path):

        model.load_state_dict(
            torch.load(
                best_model_params_path,
                map_location=device,
                weights_only=True
            )
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        eps=1e-9
    )

    scheduler_warmup = LinearLR(
        optimizer,
        total_iters=warmup_steps
    )

    scheduler_decay = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            scheduler_warmup,
            scheduler_decay
        ],
        milestones=[warmup_steps]
    )

    start_time = time.time()

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iters,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device
    )

    # Save final model
    torch.save(
        model.state_dict(),
        best_model_params_path
    )

    end_time = time.time()

    print(
        f"Training time: "
        f"{(end_time - start_time) / 60:.2f} minutes"
    )

    plt.plot(
        train_losses,
        label="train_loss"
    )

    plt.plot(
        val_losses,
        label="validation_loss"
    )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "training_validation_loss.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    return train_losses, val_losses




