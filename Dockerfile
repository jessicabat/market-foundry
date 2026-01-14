FROM zjunlp/oneke:v4

# Install git, jupyter, and Python dependencies
RUN apt-get update && apt-get install -y git jupyter \
    && pip install --no-cache-dir neo4j nltk \
    && python -m nltk.downloader punkt punkt_tab -d /usr/local/share/nltk_data \
    && ln -s $(which jupyter-notebook) /usr/local/bin/start-notebook.sh

# Ensure NLTK looks in the right place
ENV NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app
CMD ["/bin/bash"]