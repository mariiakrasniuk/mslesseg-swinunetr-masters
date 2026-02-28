import torch
import matplotlib.pyplot as plt

checkpoint = torch.load("/home/dima/projects/mslesseg-swinunetr/final_model.pth", map_location="cuda")

train_loss = checkpoint["train_loss"]
val_dice = checkpoint["val_dice"]


plt.figure(figsize=(6,4))
plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()
plt.savefig("train_loss_curve.png")
plt.show()
plt.close()

plt.figure(figsize=(6,4))
plt.plot(val_dice, label="Validation Dice", color="green")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Validation Dice Score")
plt.grid(True)
plt.legend()
plt.savefig("val_dice_curve.png")
plt.show()
