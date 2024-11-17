from system_logging import SystemLogging
import subprocess
from systemds_lr import run_systemds_lr
from pytorch_lr import run_pytorch_lr
from tensorflow_lr import run_tensorflow_lr

from systemds.context import SystemDSContext

if __name__ == "__main__":
    bucket = "gs://kalebwbishop-bucket/"

    subprocess.run([
        'gsutil', 'cp', f"{bucket}laptop_target.csv", 'laptop_target.csv'
    ])

    subprocess.run([
        'gsutil', 'cp', f"{bucket}laptop_features.csv", 'laptop_features.csv'
    ])


    # TensorFlow Logging
    tensorflow_logging = SystemLogging("tensorflow")
    tensorflow_logging.start_logging()
    run_tensorflow_lr(tensorflow_logging)
    tensorflow_logging.stop_logging()

    # PyTorch Logging
    pytorch_logging = SystemLogging("pytorch")
    pytorch_logging.start_logging()
    run_pytorch_lr(pytorch_logging)
    pytorch_logging.stop_logging()

    # SystemDS Logging
    systemds_logging = SystemLogging("systemds")
    systemds_logging.start_logging()
    run_systemds_lr(systemds_logging)
    systemds_logging.stop_logging()

    try:
        # Copy TensorFlow log to GCS
        subprocess.run([
            'gsutil', 'cp', '/tmp/local_logs/tensorflow.log', f"{bucket}logs/"
        ])
        print(f"Copied TensorFlow log to {bucket}/logs/")

        # Copy PyTorch log to GCS
        subprocess.run([
            'gsutil', 'cp', '/tmp/local_logs/pytorch.log', f"{bucket}logs/"
        ])
        print(f"Copied PyTorch log to {bucket}/logs/")
        
        # Copy SystemDS log to GCS
        subprocess.run([
            'gsutil', 'cp', '/tmp/local_logs/systemds.log', f"{bucket}logs/"
        ])
        print(f"Copied SystemDS log to {bucket}/logs/") 

    except Exception as e:
        print(f"Error copying SystemDS log to GCS: {e}")
