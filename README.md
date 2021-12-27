# Reasoning agent project: Policy Networks for Non-Markovian Reinforcement Learning Rewards

## Setup (tested on python 3.8.10 and 3.8.12)

* Install environment:

```bash
git clone https://github.com/ireneb97/RA_project.git
cd RA_project
```

* Install Lydia (before you need to [install Docker](https://www.docker.com/get-started)):

```bash
docker pull whitemech/lydia:latest
```

```bash
echo '#!/usr/bin/env sh' > lydia
echo 'docker run -v$(pwd):/home/default whitemech/lydia lydia "$@"' >> lydia
sudo chmod u+x lydia
sudo mv lydia /usr/local/bin/
```

* Create and initialize environment:
```bash
python3 -m venv ./venv
source venv/bin/activate
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

## Train an agent
In folder `config` are stored some configurations we have used. We suggest to not to change those files as they already store the best values for each map and algorithm pair. 
You can run one of them (e.g. config1.cfg) by running the command:
```bash
python3 main.py --config_file config1.cfg
```

## Test an agent
In folder `model` we saved our trained agents, you can run one of them by using this command:
```bash
python3 main.py --trained_model_path model/ppo_0
```

## Miscellaneous

- You can read in details about this project [here](https://github.com/ireneb97/RA_project/blob/main/Report%20Reasoning%20Agent.pdf), inside our report.
- You can see our video presentation [here]().
- You can see the slides we used in our presentation [here]().

## Authors

- Irene Bondanza (bondanza.1747677@studenti.uniroma1.it)
- Matteo Emanuele (emanuele.1912588@studenti.uniroma1.it)
- Pietro Manganelli Conforti (manganelliconforti.1754825@studenti.uniroma1.it)
