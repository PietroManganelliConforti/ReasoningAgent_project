# Reasoning agent project: Policy Networks for Non-Markovian Reinforcement Learning Rewards

## Setup (tested on python 3.8.10 and 3.8.12)

* Install environment:

`git clone https://github.com/ireneb97/RA_project.git`

`cd RA_project`

* Install Lydia (before you need to [install Docker](https://www.docker.com/get-started)):

`docker pull whitemech/lydia:latest`

```
echo '#!/usr/bin/env sh' > lydia
echo 'docker run -v$(pwd):/home/default whitemech/lydia lydia "$@"' >> lydia
sudo chmod u+x lydia
sudo mv lydia /usr/local/bin/
```

* Create and initialize environment:

`python3 -m venv ./venv`

`source venv/bin/activate`

* Install dependencies:

`pip install -r requirements.txt`

## Authors

- Irene Bondanza (bondanza.1747677@studenti.uniroma1.it)
- Matteo Emanuele (emanuele.1912588@studenti.uniroma1.it)
- Pietro Manganelli Conforti (manganelliconforti.1754825@studenti.uniroma1.it)
