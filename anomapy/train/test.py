def test_gpu():


    print("TESTING DEPENDANCIES:")
    import pyworld.toolkit.tools.torchutils as tu
    print("SUCCESS")

    print("TESTING DEVICE AVAILABILITY:")
    device = tu.device()
    if device == 'cuda':
        print("SUCCESS")
    else:
        print("OH NO - check GPU drivers?")
