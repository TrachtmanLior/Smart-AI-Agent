apt-get update
echo "wireshark-common wireshark-common/install-setuid boolean true" | sudo debconf-set-selections
apt-get install -y tshark
pip3 install --user -r requirements.txt