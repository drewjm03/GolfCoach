import ctypes
import sys

# load DLL
dll = ctypes.WinDLL(r"../sdk/See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620/Win32/Binary/64Bit/HIDLibraries/eCAMFwSw.dll")

# Define types
UINT8 = ctypes.c_ubyte
UINT32 = ctypes.c_uint32
BOOL = ctypes.c_bool

print("Testing DLL functions...")

# Test GetDevicesCount
print("1. Testing GetDevicesCount...")
device_count = UINT32(0)
try:
    result = dll.GetDevicesCount(ctypes.byref(device_count))
    print(f"   GetDevicesCount result: {result}")
    print(f"   Device count: {device_count.value}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

if device_count.value == 0:
    print("   No devices found!")
    sys.exit(1)

print("2. Testing GetDevicePaths...")
try:
    # Allocate memory for device paths
    device_paths = (ctypes.POINTER(ctypes.c_char) * device_count.value)()
    for i in range(device_count.value):
        device_paths[i] = ctypes.cast(ctypes.create_string_buffer(260), ctypes.POINTER(ctypes.c_char))
    
    result = dll.GetDevicePaths(device_paths)
    print(f"   GetDevicePaths result: {result}")
    
    if result:
        for i in range(device_count.value):
            path_str = ctypes.string_at(device_paths[i])
            print(f"   Device {i}: {path_str}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("3. Testing InitExtensionUnit...")
try:
    device_path = device_paths[0]
    result = dll.InitExtensionUnit(device_path)
    print(f"   InitExtensionUnit result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("4. Testing SetStreamMode24CUG...")
try:
    dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]
    dll.SetStreamMode24CUG.restype = BOOL
    
    result = dll.SetStreamMode24CUG(UINT8(0x00), UINT8(0x01))
    print(f"   SetStreamMode24CUG result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("5. Testing DeinitExtensionUnit...")
try:
    result = dll.DeinitExtensionUnit()
    print(f"   DeinitExtensionUnit result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")

print("All tests completed!")
