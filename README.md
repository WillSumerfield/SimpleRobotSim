# SimpleRobotSim

A simple robotics simulator using PyMunk.

### **Version:**

Created with Python version `3.10.12`

## How to Use

Run `python main.py`, with any of the following arugments.
*Ex: python src/main.py manual --agent-save-file "myAgent.pkl"*

* **Required: mode**
  * manual - Play the environment yourself
  * train - Train a new agent on the environment
  * test - Test the saved model on the environment
* **Optional: --agent-save-file**
  * **When Training:** The name of the file to save the agent to. Can be used to avoid overwriting another agent save file. Defaults to *agent.pkl*
  * **When Testing:** The name of the agent save file to read from. Defaults to *agent.pkl*
