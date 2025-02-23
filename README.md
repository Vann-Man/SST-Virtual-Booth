Filter Setup Instructions
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

### Using the Application
Once the application is running, you can interact with it using the terminal menu:

1. **Activate/Deactivate Filters:**
   - **Activate Filter**: Enter `1` to apply the selected prop(s) to detected faces.
   - **Deactivate Filter**: Enter `2` to remove the prop(s) from the video stream.

2. **Select Props:**
   - Enter `3` to choose props to overlay on faces. You can select multiple props by entering numbers separated by spaces:
     - `1` for Hat
     - `2` for Heart Sunglasses

3. **Activate/Deactivate Backgrounds:**
   - **Activate Background**: Enter `4` to apply the selected background.
   - **Deactivate Background**: Enter `5` to remove the background.

4. **Select Backgrounds:**
   - Enter `6` to choose a background:
     - `1` for Zoom
     - `2` for SST

5. **Exit the Application:**
   - Enter `7` to exit the application.
   - Alternatively, press the `Esc` key to exit at any time.


## Contact
For questions or support, contact ganguly_rishav@s2022.ssts.edu.sg,pok_vann_xyn@s2022.ssts.edu.sg,sim_jie_yi_kieren@s2022.ssts.edu.sg.