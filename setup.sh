source /home/jovyan/.bashrc
grep -v '^#' requirements.txt | xargs -n 1 -L 1 pip install --default-timeout=100 --no-cache-dir