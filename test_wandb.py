import wandb
import random
import time

# Force W&B to strictly look for system metrics during init
run = wandb.init(
    project="bitnet-test",
    name="system-hardware-test"
)

print("🚀 Script started! Forcing it to run for 2 minutes to collect GPU/CPU metrics...")

# Loop 120 times with a 1-second pause = 120 seconds total
for epoch in range(120):
    dummy_loss = 2.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
    wandb.log({"loss": dummy_loss, "epoch": epoch})
    
    if epoch % 10 == 0:
        print(f"⏳ Progress: {epoch}/120 seconds elapsed...")
        
    time.sleep(1.0) 

wandb.finish()
print("🏁 Done! Check your dashboard now.")