# Stock-Prediction-using-Deep-Reinforcement-Learning
Using Deep Learning alongside Reinforcement Learning to estimate stock price trends

Check [requirements.txt](requirements.txt) file to setup the environment.

We use Temporal Difference (TD) algorithm estimate the state value of one timestep. Using Deep Neural Networks,
we estimate ```V<sub>t+1</sub>(s)``` used in TD update step.

To run the code, execute
```
python Game.py -a <alpha> -e <episodes> -g <gamma>
```

Link to the paper Implemented [Stock Price Prediction using Reinforcement Learning](https://ieeexplore.ieee.org/document/931880)
