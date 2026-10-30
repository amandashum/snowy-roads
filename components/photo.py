"""Keep workspace previews compact while preserving the original photo."""

import io

from PIL import Image, ImageOps
import streamlit as st


def preview_image(source, max_height=240):
    """Return an oriented thumbnail without changing the source image or upload."""
    if hasattr(source, "getvalue"):
        content = source.getvalue()
        if len(content) > 10 * 1024 * 1024:
            raise ValueError("Choose a photo smaller than 10 MB.")
        source = io.BytesIO(content)
    with Image.open(source) as original:
        if original.format not in {"JPEG", "PNG"}:
            raise ValueError("Choose a JPEG or PNG photo.")
        if original.width * original.height > 20_000_000:
            raise ValueError("Choose a photo with at most 20 million pixels.")
        preview = ImageOps.exif_transpose(original).convert("RGB")
        preview.thumbnail((560, max_height), Image.Resampling.LANCZOS)
    return preview


@st.dialog("Road Photo", width="large", on_dismiss="rerun")
def show_full_photo(source):
    """Open the original photo separately from the workspace controls."""
    if hasattr(source, "getvalue"):
        source = source.getvalue()
    st.image(source, alt="Full road photo", width="stretch")
    if st.button("Done", key="close_full_photo", type="primary", width="stretch"):
        st.rerun()


def render_photo(source, key, *, max_height=240):
    """Render a bounded thumbnail with an optional full-photo dialog."""
    try:
        preview = preview_image(source, max_height=max_height)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f"This photo could not be previewed. {error}")
        return False
    with st.container(key=f"preview_{key}"):
        st.image(preview, alt="Road photo preview", width="content")
    if st.button("Open Full Photo", key=f"full_photo_{key}", icon=":material/open_in_full:"):
        show_full_photo(source)
    return True
