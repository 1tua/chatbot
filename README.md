# Squirro Bot
Squirro Bot is a chatbot for answering Squirro queries (FAQ)

#How It Works
1.	CB.py contains the neural network that trains the chatbot on questions and answers. intents.json contains the questions and answers.
2.	Chatbot_GUI uses the model trained in CB.py and displays a Graphic User Interface (GUI) where users can ask their questions to the chatbot amd receive responses.
3.	chatbot_logs.log contains a log of questions users ask that the chatbot hasn't been trained on. 

# Installation
Use the package manager pip to install tensorflow, pickle, nltk

```bash
pip install tensorflow
pip install pickle
pip install nltk
```
For nltk libary to work "nltk_data" folder should be found in the following path ~/Squirro_bot/.venv/nltk_data

# Requirements
The software requirements for this project are:

```bash
Python 3.6
Microsoft Visual Studio Code 1.43
```

# Usage

- Download the lastest code from TFS repo - Squirro_Bot 
- Open Squirro_Bot folder in Visual Studio Code 
- Run Chatbot_GUI.py file in Terminal 
- Squirro Bot opens allowing users to type in their queries.

```python
python Chatbot_GUI.py'
```


