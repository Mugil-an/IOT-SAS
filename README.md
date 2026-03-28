# IoT SAS Project Setup

## Prerequisites
- Python 3.10 (recommended)
- Git (optional, for version control)

## Setting up the Project Environment

1. **Clone the repository** (if using version control):
   ```sh
   git clone <your-repo-url>
   cd Iot_sas
   ```

2. **Create a virtual environment (venv):**
   ```sh
   python -m venv venv310
   ```

3. **Activate the virtual environment:**
   - On **Windows** (Command Prompt):
     ```sh
     venv310\Scripts\activate.bat
     ```
   - On **Windows** (PowerShell):
     ```sh
     venv310\Scripts\Activate.ps1
     ```
   - On **macOS/Linux**:
     ```sh
     source venv310/bin/activate
     ```

4. **Upgrade pip (recommended):**
   ```sh
   python -m pip install --upgrade pip
   ```

5. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

- To run the main app:
  ```sh
  streamlit run app.py
  ```

- To run the watcher script:
  ```sh
  python watcher.py
  ```

## Notes
- Always activate the virtual environment before running scripts or installing packages.
- If you encounter issues with package installation, ensure you are using Python 3.10 and the virtual environment is active.
