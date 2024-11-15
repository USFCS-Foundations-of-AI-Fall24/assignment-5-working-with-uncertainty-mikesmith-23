import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.


class Sequence:

    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# HMM model
class HMM:

    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """Reads HMM structure from transition (basename.trans) and emission (basename.emit) files, 
        storing probabilities in self.transitions and self.emissions."""

        # Load transitions from basename.trans file
        self.transitions = {}
        with open(f"{basename}.trans", 'r') as trans_file:
            for line in trans_file:
                parts = line.strip().split()
                if len(parts) == 3:
                    source_state, target_state, prob = parts
                    self.transitions.setdefault(source_state,
                                                {})[target_state] = float(prob)

        # Load emissions from basename.emit file
        self.emissions = {}
        with open(f"{basename}.emit", 'r') as emit_file:
            for line in emit_file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, observation, prob = parts
                    self.emissions.setdefault(state,
                                              {})[observation] = float(prob)


## you do this.

    def generate(self, n):
        """Generate an n-length Sequence by randomly sampling from the HMM."""

        current_state = "#"  # Start @ initial state
        states = []
        observations = []

        for _ in range(n):
            # Choose the next state based on transition probabilities
            next_states = list(self.transitions[current_state].keys())
            next_probs = list(self.transitions[current_state].values())
            current_state = random.choices(next_states, weights=next_probs)[0]
            states.append(current_state)

            # Stop if we reach a lander's terminal state
            if current_state == "5,5":
                break

            # Choose an observation based on emission probabilities
            observation_choices = list(self.emissions[current_state].keys())
            emission_probs = list(self.emissions[current_state].values())
            observation = random.choices(observation_choices,
                                         weights=emission_probs)[0]
            observations.append(observation)

        return Sequence(states, observations)

    def forward(self, sequence):
        pass

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

    def viterbi(self, observations):
        pass

    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

if __name__ == "__main__":

    # python3 HMM.py cat --generate 20
    # python3 HMM.py lander --generate 20
    # python3 HMM.py partofspeech --generate 20

    parser = argparse.ArgumentParser()
    parser.add_argument("basename", type=str)
    parser.add_argument("--generate", type=int)

    args = parser.parse_args()

    # Create an HMM instance and load the model
    hmm = HMM()
    hmm.load(args.basename)

    # Generate and print a random sequence
    if args.generate:
        sequence = hmm.generate(args.generate)
        print(sequence)
