#!/bin/bash
jupyter notebook \
    --allow-root \
    --no-browser \
    --port=8888 \
    --NotebookApp.allow_remote_access=True
