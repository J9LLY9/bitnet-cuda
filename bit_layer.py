import torch
import torch.nn as nn

class BitLinear(nn.Linear):
    """
    This is a 'prestige' layer. 
    It replaces standard 32-bit math with Ternary (1.58-bit) math.
    """
    def forward(self, x):
        # 1. Get our weights (the knobs)
        w = self.weight
        
        # 2. Calculate 'Gamma' (The scaling factor from the Microsoft Paper)
        # This keeps our numbers from getting too small/large.
        gamma = w.abs().mean()
        
        # 3. THE STE TRICK (The 'Magic' line)
        # We round for the guess, but 'detach' the rounding from the learning math.
        w_quant = (w / (gamma + 1e-5)).round().clamp(-1, 1)
        w_final = w + (w_quant - w).detach() 

        # 4. Run the actual math using our 'Bit' weights
        # We use F.linear because we are overriding the standard behavior
        return torch.nn.functional.linear(x, w_final, self.bias)

# --- TEST IT ---
# Create a layer with 10 inputs and 5 outputs
layer = BitLinear(10, 5)

# Give it some random data
input_data = torch.randn(1, 10)
output = layer(input_data)

print("Output from your BitLinear layer:")
print(output)
print("\nCheck the weights - they are still decimals, but the MATH used -1, 0, 1!")