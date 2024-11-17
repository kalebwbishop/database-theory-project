tar -a -c -f support.zip system_logging.py systemds_lr.py tensorflow_lr.py pytorch_lr.py
gsutil cp support.zip gs://kalebwbishop-bucket/scripts/support.zip
gsutil cp main.py gs://kalebwbishop-bucket/scripts/main.py
