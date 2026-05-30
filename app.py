import os
import sys

# Ensure root directory is in the path for imports
sys.path.insert(0, os.path.dirname(__file__))

from app.web_app import create_app

# Create the Gradio demo application
app = create_app()

if __name__ == "__main__":
    # Launch with public access bindings (standard for hosting and local run)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
