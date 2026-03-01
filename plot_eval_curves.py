import argparse
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default="baseline",
                    help="Prefix used when saving training_history_<run_name>.pth")
args = parser.parse_args()

checkpoint = torch.load(f"training_history_{args.run_name}.pth", map_location="cpu")

train_loss = checkpoint["train_loss"]
val_loss   = checkpoint["val_loss"]
train_dice = checkpoint["train_dice"]
val_dice   = checkpoint["val_dice"]


plt.figure(figsize=(6, 4))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss — {args.run_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"loss_curves_{args.run_name}.png")
plt.show()
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(train_dice, label="Train Dice")
plt.plot(val_dice,   label="Validation Dice", color="green")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title(f"Validation Dice Score — {args.run_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"val_dice_curve_{args.run_name}.png")
plt.show()
plt.close()
