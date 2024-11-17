# database-theory-project

Command to upload file to could bucket
gcutil cp <LOCAL_FILE_PATH> gs://kalebwbishop-bucket/<CLOUD_FILE_PATH>

Command to run job
gcloud dataproc jobs submit pyspark gs://kalebwbishop-bucket/scripts/main.py --cluster=cluster-1 --region=us-east1 --py-files=gs://kalebwbishop-bucket/scripts/support.zip

Command to create cluster
gcloud dataproc clusters create <CLUSTER_NAME> --region=<REGION> --initialization-actions=gs://kalebwbishop-bucket/scripts/requirements-installer.sh

gcloud dataproc clusters create cluster-1 --region=us-east1 --num-workers=2 --master-machine-type=n1-standard-2 --worker-machine-type=n1-standard-2 --image-version=2.0-debian10 --master-boot-disk-size=30GB --worker-boot-disk-size=30GB --initialization-actions=gs://kalebwbishop-bucket/scripts/requirements-installer.sh 

