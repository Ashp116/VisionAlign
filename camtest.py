from pypylon import pylon

tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()

if not devices:
    print("No devices found.")
else:
    for i, dev in enumerate(devices):
        print(f"[{i}] {dev.GetFriendlyName()} - {dev.GetIpAddress()}")

    # Open the first device
    camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
    camera.Open()
    print("Opened:", camera.GetDeviceInfo().GetModelName())
    camera.Close()
