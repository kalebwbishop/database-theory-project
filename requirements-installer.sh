# requirements-installer.sh
#!/bin/bash
apt-get update
apt-get install -y python3-pip
gsutil cp gs://kalebwbishop-bucket/requirements.txt /tmp/requirements.txt
pip3 install -r /tmp/requirements.txt

# Install Java 11
wget https://download.java.net/java/ga/jdk11/openjdk-11_linux-x64_bin.tar.gz
sudo tar -xvf openjdk-11_linux-x64_bin.tar.gz -C /opt

# Set JAVA_HOME
echo "export JAVA_HOME=/opt/jdk-11" | sudo tee -a /etc/profile.d/java.sh
echo "export PATH=\$JAVA_HOME/bin:\$PATH" | sudo tee -a /etc/profile.d/java.sh
sudo chmod +x /etc/profile.d/java.sh
source /etc/profile.d/java.sh
