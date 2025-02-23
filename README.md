Sunglasses Thingamajig Setup Instructions
=========================================

This document provides instructions to set up a virtual environment and install the necessary dependencies for the Sunglasses Thingamajig project.

1. Create a Virtual Environment
-------------------------------
Navigate to your project directory and run the following command to create a virtual environment:

    python -m venv venv

2. Activate the Virtual Environment
-----------------------------------
- On Windows, run:

      venv\Scripts\activate

- On macOS and Linux, run:

      source venv/bin/activate

3. Install the Requirements
---------------------------
With the virtual environment activated, install the required packages using:

    pip install -r requirements.txt

4. Deactivate the Virtual Environment
-------------------------------------
When finished, deactivate the virtual environment by running:

    deactivate

These steps will ensure that your project dependencies are installed in an isolated environment, preventing conflicts with other projects or system-wide packages.

## Running the Code
To run the `cv.py` script, ensure your virtual environment is activated and execute the following command:

```bash
python cv.py
```

### Interacting with the Application
- Use the menu in the terminal to activate or deactivate filters and backgrounds.
- Press the `Esc` key to exit the application at any time.

## License
This project uses the following license: [MIT License](link-to-license).

## Contact
For questions or support, contact [your-email@example.com].