# *DataPal*: The Ultimate Data Science Platform

## Quick Local Setup

Embark on your journey with the greatest data science platform in just a few simple steps.

### Prerequisites:
- **Visual Studio Code**: Ensure VS Code is installed on your system.
- **Dev Containers Extension**: Make sure to have the Dev Containers extension installed in VS Code.
- **Docker**: Docker should be installed on your local machine.

### Setting Up:
1. **Clone the Repository**: Start by cloning the repository to your local machine.
2. **Open in Dev Container**:
   - VS Code should automatically prompt you to reopen the project in a Dev Container.
   - Make sure the Docker engine is running!
   - If you don't see the prompt, press `Ctrl/Cmd + Shift + P` to open the command palette, then search for "Dev-Containers: Reopen in Container" and select it.

Your local setup is now primed and ready.

## Run the Server

Get your data science platform up and running with these steps:

1. Open a terminal in VS Code.
2. Execute `uvicorn app.main:app --reload`. This command will start the server with live reloading, streamlining your development workflow.
3. Go to `http://127.0.0.1:8000/docs` to start testing the back-end!

---

Dive into data science with *DataPal*!
