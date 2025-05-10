## Intro to AI – Final Project: DrawGuess with AI Judge

### Introduction

In this project, we integrate an AI judging system into the original DrawGuess game. The AI model will act as the question master by selecting a target image from a predefined pool. Players will attempt to draw images that resemble the given prompt, and the AI will evaluate which player's drawing is closest to the original.

We plan to implement multiple AI models as judges to compare their performance and determine which one is most effective.


> Dataset – Currently, we are using hand-drawn images as a temporary dataset.

### Execute

- Environment Set up - Python 3.11.2
```
pip install -r requirements.txt
```

- Run the server
```
python server.py
```

- Run the client
```
python client.py
```