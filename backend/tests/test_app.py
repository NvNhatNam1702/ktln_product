import pytest
from fastapi.testclient import TestClient
import os
import shutil
from unittest.mock import MagicMock

# Add backend directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, ai_engine

# Use TestClient for making requests to the app
client = TestClient(app)

# --- Mocking PixelNeRFWrapper ---
# This avoids running the actual model which is slow and has dependencies (checkpoints)

@pytest.fixture
def mock_render_video(monkeypatch):
    """
    Mock the render_video method of the PixelNeRFWrapper instance.
    This fixture will replace the real method with a dummy one for the duration of a test.
    """
    # Create a dummy output file that the mock function will return
    dummy_output_dir = "test_output"
    dummy_output_filename = "spin.gif"
    dummy_output_path = os.path.join(dummy_output_dir, dummy_output_filename)

    os.makedirs(dummy_output_dir, exist_ok=True)
    with open(dummy_output_path, "w") as f:
        f.write("GIF") # dummy content

    def mock_return(image_path, gif=True):
        return dummy_output_path

    # The 'monkeypatch' fixture is a feature of pytest
    monkeypatch.setattr(ai_engine, "render_video", MagicMock(side_effect=mock_return))
    
    yield # Test runs here

    # Teardown: clean up dummy files and directories
    if os.path.exists(dummy_output_dir):
        shutil.rmtree(dummy_output_dir)
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")


def test_reconstruct_endpoint(mock_render_video):
    """
    Test the /reconstruct endpoint with a mocked AI engine.
    """
    # Create a dummy file to upload
    dummy_image_content = b"fake image data"
    files = {"file": ("test_image.png", dummy_image_content, "image/png")}

    # Make the request
    response = client.post("/reconstruct", files=files)

    # Assertions
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/gif'
    # Check that the content is what we wrote in our dummy gif
    assert response.content == b"GIF"

    # Verify that the ai_engine.render_video was called
    ai_engine.render_video.assert_called_once()
    # Check the path of the saved upload
    ai_engine.render_video.assert_called_with("uploads/test_image.png", gif=True)
