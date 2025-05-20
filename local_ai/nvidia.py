import pynvml

class NvidiaGPUManager:

    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.devices = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self.devices.append(pynvml.nvmlDeviceGetName(handle))

    def __del__(self):
        pynvml.nvmlShutdown()
            
    def get_device_count(self):
        return self.device_count
    
    def get_devices(self):
        return self.devices
    
    def get_free_vram(self):
        free_vram = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_vram_gb = info.free / 1024 / 1024 / 1024
            free_vram.append(free_vram_gb)
        return free_vram
    
    def get_idle_gpus(self):
        idle_gpus = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError_NotSupported:
                idle_gpus.append(True)
                continue
            idle_gpus.append(len(info) == 0)
        return idle_gpus
    
    def get_available_gpus(self):
        available_gpus = []
        for i in range(self.device_count):
            if self.get_idle_gpus()[i]:
                available_gpus.append(i)
        return available_gpus
        
    def total_vram_gpus(self, gpu_indices: list[int]):
        total_vram = 0
        for i in gpu_indices:
            total_vram += self.get_free_vram()[i]
        return total_vram
    
if __name__ == "__main__":
    manager = NvidiaGPUManager()
    print(manager.get_device_count())
    print(manager.get_devices())
    print(manager.get_free_vram())
    print(manager.get_idle_gpus())
    available_gpus = manager.get_available_gpus()
    print(available_gpus)
    print(manager.total_vram_gpus(available_gpus))