import psutil
import time

def system_logging(file_name: str, output = None):
    if output:
        with open(f"./{file_name}.log", "a") as f:
                f.write(f"\n{output}")
    else:
        with open(f"./{file_name}.log", "w") as f:
            f.write("System Usage Log\n")

        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')

            with open(f"./{file_name}.log", "a") as f:
                f.write(f"System Usage - CPU: {cpu_usage}%, Memory: {memory_info.percent}%, Disk: {disk_info.percent}%\n")
            