![](images/original-e8c642f35996f414749c5fdea2ce991b.png)

## TODOs

1. [ ] https://www.kaggle.com/competitions/llm-20-questions/discussion/519723
2. [ ] https://www.kaggle.com/competitions/llm-20-questions/discussion/513394
3. [ ] https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Guess_the_word.ipynb
4. [ ] https://www.kaggle.com/code/guillaumegilles/walkthrough-kaggle-starter-notebook/edit
5. [ ] https://www.kaggle.com/code/lohmaa/llm20-agent-alpha
       https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750

[Run/Debug LLM 20 Questions in a Notebook](https://www.kaggle.com/code/rturley/run-debug-llm-20-questions-in-a-notebook)

## Running LLM 20 Questions (default)

You can then create and run the game in this environment. When you run the game, you must submit a list of four agents:

    "Agent1" (guesser for Team 1),
    "Agent2" (answerer for Team 1),
    "Agent3" (guesser for Team 2),
    "Agent4" (answerer for Team 2).

In the competition, you are randomly paired with a teammate to either be the guesser or the answerer.

(When I first started this competition, I mistakenly thought your agent plays both the guesser and answerer role for the team. But you are paired with someone else in the competition. You do well or poorly depending on your ability to cooperate with a random partner.)

```python
%%time
game_output = env.run(agents=[simple_agent1, simple_agent2, simple_agent3, simple_agent4])
```

The game in this example completes quickly since the simple agents respond immediately. A real game with large LLM's as agents could take a minute for each step, so the total game could take an hour!

You can look at the data from each step of the game in game_output.

If want to watch the game visually, you can render it.

```python
env.render(mode="ipython", width=400, height=400)
```

## Create an Agent that Could be Submitted¶

To submit an agent to the competition, you need to write the Python code for the agent in a file titled main.py and put it along with any supporting files in submission.tar.gz

A simple example is below. Of course, in the actual competition, you'll probably want to use a real LLM like in the official starter notebook (https://www.kaggle.com/code/ryanholbrook/llm-20-questions-starter-notebook). Running LLM agents in a notebook will take more time and memory, so if you're testing your LLM agent as player 1, you might want to put a simple agent as player 2.

    Create a directory /kaggle/working/submission with a subdirectory lib where you would put any supporting files

```python
import os
submission_directory = "/kaggle/working/submission"
submission_subdirectory = "lib"
# Create the main directory if it doesn't exist
if not os.path.exists(submission_directory):
    os.mkdir(submission_directory)
    subdirectory_path = os.path.join(submission_directory, submission_subdirectory)
    os.mkdir(subdirectory_path)
```

```python
# create an example file to save in the lib directory
import csv
with open(os.path.join(subdirectory_path, "example.csv"),mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["cow", "horse"])
```

- Write the main.py Python code for your agent
- The environment will use the last function in main.py for your agent, in this case agent_fun()

```python
%%writefile /kaggle/working/submission/main.py

import os
import sys
import csv
import random



# If you put other files (e.g. model weights) in your submission/lib directory, you need to set the path
KAGGLE_COMPETITION_PATH = "/kaggle_simulations/agent/" # competition path
if os.path.exists(KAGGLE_COMPETITION_PATH):  # if running in the competition
    subdirectory_path = os.path.join(KAGGLE_COMPETITION_PATH, "lib")
else: # if running in notebook
    subdirectory_path = os.path.join("/kaggle/working/submission/", "lib")
sys.path.insert(0, subdirectory_path)


# Loading our example file
with open(os.path.join(subdirectory_path,"example.csv"), mode='r') as file:
    reader = csv.reader(file)
    guess_list = list(reader)
    guess_list = guess_list[0]

# Setting a random "animal" from example file as a global variable
animal = random.choice(guess_list)

# Last function in the main.py will be the agent function
def agent_fn(obs, cfg):

    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        response = f'Does it look like a {animal}?'
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        if obs.answers[-1]=="yes":
            response = animal
        else:
            response = "penguin"
    # if agent is the answerer
    elif obs.turnType == "answer":
        if obs.keyword in obs.questions[-1]:
            response = "yes"
        else:
            response = "no"

    return response
```

This main.py file with the agent is ready to submit along with the /lib/example.csv supporting file.

```bash
!apt install pigz pv > /dev/null
!tar --use-compress-program='pigz --fast --recursive | pv' -cf submission.tar.gz -C /kaggle/working/submission .
```

You can run the agent in main.py from this Jupyter notebook as both players on Team 1, and we'll use simple_agent3 and simple_agent4 for Team 2.

```python
game_output = env.run(agents=["/kaggle/working/submission/main.py", "/kaggle/working/submission/main.py", simple_agent3, simple_agent4])
env.render(mode="ipython", width=400, height=400)
```

## Debugging Tips

When you're designing and debugging, you normally want to change some of the optional arguments in creating the environment. These include:

env = make(environment, configuration=None, info=None, steps=None, logs=None, debug=False, state=None)

You can look at the specifications in env.specification to learn about how configuration and other objects defined in the environment. It has explanations and shows the default values.

When working on new agents, I'd suggest changing the configuration to run a shorter episode with only a few steps and setting debug=True so you can see any verbose output printed by your agents.

Here is a new environment that is better for debugging.

```python
# For debugging, play game with only two rounds
debug_config = {'episodeSteps': 7,     # initial step plus 3 steps per round (ask/answer/guess)
                'actTimeout': 5,       # agent time per round in seconds; default is 60
                'runTimeout': 60,      # max time for the episode in seconds; default is 1200
                'agentTimeout': 3600}  # obsolete field; default is 3600

env = make("llm_20_questions", configuration=debug_config, debug=True)
```

And we can have our agent print some information for debugging. I added print
statements to simple agent 1 to show what information is available in `obs`

```python
def simple_verbose_agent1(obs, cfg):

    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        response = "Is it a duck?"
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        response = "duck"
    # if agent is the answerer
    elif obs.turnType == "answer":
        response = "no"

    # Print debugging information
    print("====================")
    print(f"step = {obs.step}")
    print(f"turnType = {obs.turnType}")
    print("obs =")
    print(obs)
    print(" ")
    print(f'response = "{response}"')

    return response
```

Putting this simple_verbose_agent1 as both players on Team 1 allows us to observe
each of the three turn types (ask/guess/answer).

```python
game_output = env.run(agents=[simple_verbose_agent1,simple_verbose_agent1, simple_agent3, "/kaggle/working/submission/main.py"])
```

---

---

---

# LLM 20 Questions

In this simulation competition, you must create a language model capable of playing the game 20 Questions. Teams will be paired in 2 vs 2 player matchups and race to deduce the secret word first.

## Description

Is it a person, place or thing? Is it smaller than a bread box? Is it smaller than a 70B parameter model? 20 Questions is an age-old deduction game where you try to guess a secret word in twenty questions or fewer, using only yes-or-no questions. Players try to deduce the word by narrowing their questions from general to specific, in hopes of guessing the word in the fewest number of questions.

Each team will consist of one guesser LLM, responsible for asking questions and making guesses, and one answerer LLM, responsible for responding with "yes" or "no" answers. Through strategic questioning and answering, the goal is for the guesser to correctly identify the secret word in as few rounds as possible.

This competition will evaluate LLMs on key skills like deductive reasoning, efficient information gathering through targeted questioning, and collaboration between paired agents. It also presents a constrained setting requiring creativity and strategy with a limited number of guesses.Success will demonstrate LLMs' capacity for not just answering questions, but also asking insightful questions, performing logical inference, and quickly narrowing down possibilities.

## Setting up the environment

1.

## Evaluation

Each day your team is able to submit up to 5 agents (bots) to the competition. Each submission will play episodes (games) against other bots on the leaderboard that have a similar skill rating. Over time, skill ratings will go up with wins, down with losses, or evened out with ties.

This competition is configured to run in a cooperative, 2 vs. 2 format. Your bot will be randomly paired with a bot of similar skill in order to face off against another random pairing. On each pair, one bot will be randomly assigned as questioner and the other as answerer. Since you win/lose/tie as a pair, you are incentivized to work together!

Every bot submitted will continue to play episodes until the end of the competition, with newer bots selected to play more frequently. On the leaderboard, only your best scoring bot will be shown, but you can track the progress of all of your submissions on your Submissions page.

Each submission has an estimated skill rating which is modeled by a Gaussian $N(μ,σ2)$ where $μ$ is the estimated skill and $σ$ represents the uncertainty of that estimate which will decrease over time.

When you upload a submission, we first play a validation episode where that submission plays against copies of itself to make sure it works properly. If the episode fails, the submission is marked as error and you can download the agent logs to help figure out why. Otherwise, we initialize the submission with $μ0 = 600$ and it joins the pool of for ongoing evaluation.

### Ranking System

After an episode finishes, we'll update the rating estimate for all bots in the episode. If one bot pair won, we'll increase their $μ$ and decrease the opponent's $μ$ -- if the result was a tie, then we'll move the $μ$ values closer towards their mean. The updates will have magnitude relative to the deviation from the expected result based on the previous $μ$ values, and also relative to each bot’s uncertainty $σ$. We also reduce the σ terms relative to the amount of information gained by the result. The score by which your bot wins or loses an episode does not affect the skill rating updates.

### Final Evaluation

At the submission deadline on August 13, 2024, submissions will be locked. From August 13, 2024 to August 27th, 2024 we will continue to run episodes against a new set of unpublished, secret words. At the conclusion of this period, the leaderboard is final.

## Timeline

- May 15, 2024 - Start Date.
- August 6, 2024 - Entry Deadline. You must accept the competition rules before this date in order to compete.
- August 6, 2024 - Team Merger Deadline. This is the last day participants may join or merge teams.
- August 13, 2024 - Final Submission Deadline.
- August 13, 2024- August 27, 2024 - Estimated range when final games are played.
- August 28, 2024 - Winners announced.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## 20 Questions Rules

The game will proceed in rounds, with 20 total rounds. At the start of each round, the two questioners will each submit a question trying to guess the target word, and then submit their guess for what the target word is.

If either questioner guesses the target word correctly in that round, their team wins the game immediately. If both questioners guess correctly in the same round, then the round is a tie.

If neither questioner guesses correctly, the game continues to the next round. The two answerers will each respond to the question from the team's questioner with either "yes" or "no". Using this information, the questioners will then submit new questions and guesses in the following round.

This process repeats for up to 20 total rounds. If neither team has guessed the word after 20 rounds, the game results in a tie. The goal is for each team's questioner to guess the target word in as few rounds as possible based on the information provided by the answering agent.

`country`, `city`, `landmark`

There will be a few changes to keywords.py (this is the list of possible words to guess for the game).

Categories will be simplified into `person`, `place`, or `thing`.

Change will happen next week (first week of June)

Half way through the competition more words will be added

As stated in the rules, after the FINAL submission deadline, the words will be swapped for a set that is NOT accessible by your agents. This word set will have the same 3 categories

IMPORTANT: Do not rely on knowing the full list of possible words ahead of time!

## Timeouts, Limits and Penalties

- Questions are limited to 2000 characters
- Guesses are limited to 100 characters
- Timeouts
  - Agents are given 60 seconds per round to answer
  - Agents have an additional 300 overage seconds to use throughout the game
  - Any agent timing out will cause the game to end
- Any answering agent responding with anything other than yes or no will result in the game ending and them losing the match.

## Technical Specifications

- 100 GB of disk space
- 16 GB of RAM
- 1 T4 GPU
