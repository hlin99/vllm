BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# Normal Mode
python proxy_server.py --port 8868 -m /workspace/Llama-3.1-8B-Instruct/ -p 10.239.11.171:8866 -d 10.239.11.165:8866 --bypass-proxy
