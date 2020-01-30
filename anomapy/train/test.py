print("TESTING DEPENDANCIES:")
import pyworld.toolkit.tools.torchutils as tu
print("SUCCESS")

print("TESTING DEVICE AVAILABILITY:")
device = tu.device()
if device == 'cuda':
    print("SUCCESS")
print("OH NO - check GPU drivers?")
