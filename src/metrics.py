import psutil

def get_ram_usage_gb():
    process = psutil.Process()
    memory_info = process.memory_info()

    return memory_info.rss / (1024 ** 3)