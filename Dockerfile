FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

# Copy environment file first (better caching)
COPY environment.yml .

# Create environment, initialize conda for bash, and auto-activate environment
RUN conda env create -f environment.yml && \
  conda init bash && \
  echo "conda activate market-foundry" >> ~/.bashrc

# Start interactive shell with env activated
CMD ["/bin/bash", "--login"]
