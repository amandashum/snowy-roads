"""Cache the local inference model for dashboard submissions."""

import streamlit as st


@st.cache_resource(max_entries=2)
def cached_checkpoint(path, modified_ns, size):
    """Reuse a CPU model until its checkpoint modification time or size changes."""
    from inference import load_checkpoint
    return load_checkpoint(path)


def classify_photo(image_path, model_path):
    """Classify a photo with a cached, non-smoke-test model."""
    from inference import predict
    stat = model_path.stat()
    loaded = cached_checkpoint(str(model_path), stat.st_mtime_ns, stat.st_size)
    if loaded[1].get("smoke_test"):
        raise ValueError("Smoke-test checkpoints cannot power real report suggestions")
    return predict(image_path, model_path, loaded_checkpoint=loaded)
