## Intro to AI – Final Project: DrawGuess with AI Judger

### Introduction

In this project, we integrate an AI judging system into the original DrawGuess game. The AI model will act as the question master by selecting a target image from a predefined pool. Players will attempt to draw images that resemble the given prompt, and the AI will evaluate which player's drawing is closest to the original.

We plan to implement multiple AI models as judges to compare their performance and determine which one is most effective.

> [Dataset](https://www.kaggle.com/datasets/ankitsheoran23/sketch-to-image)  
> We pick 8 classes from `sketch/tx_000000001110`


> [!NOTE]  
> Our Environment: `Python 3.10.11` on `Windows 11`   
> If you wnat to run the client on different PC. Then modify the `IP` parameter in `server.py` and `classes.py`(for client)

### Execute

- Python venv (Optional)
```shell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- Environment Set up
```shell
pip install -r requirements.txt
```

- Run the server
```shell
python server.py
```

- Run the client
```shell
python client.py
```

### Train your own model

- [Teachable](https://teachablemachine.withgoogle.com/train)

- CNN
```shell
python trainCNN.py
```