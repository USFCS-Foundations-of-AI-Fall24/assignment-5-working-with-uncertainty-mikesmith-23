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

    def initialize_probabilities(self,
                                 observations,
                                 init_matrix,
                                 init_backtrack=None):
        """Helper for initializing probabilities in both forward and Viterbi."""
        hidden_states = [
            state for state in self.transitions.keys() if state != "#"
        ]
        for state in hidden_states:
            if "#" in self.transitions and state in self.transitions["#"]:
                init_prob = self.transitions["#"][state]
                emit_prob = self.emissions[state].get(observations[0], 0)
                init_matrix[state][0] = init_prob * emit_prob
                if init_backtrack is not None:
                    init_backtrack[state][0] = None
        return hidden_states

    def forward(self, observations):
        num_obs = len(observations)
        forward_matrix = {
            state: [0] * num_obs
            for state in self.transitions if state != "#"
        }
        hidden_states = self.initialize_probabilities(observations,
                                                      forward_matrix)

        for obs_index in range(1, num_obs):
            for current_state in hidden_states:
                total_prob = sum(
                    forward_matrix[prev_state][obs_index - 1] *
                    self.transitions[prev_state].get(current_state, 0)
                    for prev_state in hidden_states)
                forward_matrix[current_state][
                    obs_index] = total_prob * self.emissions[
                        current_state].get(observations[obs_index], 0)

        final_probs = {
            state: forward_matrix[state][num_obs - 1]
            for state in hidden_states
        }
        most_probable_final_state = max(final_probs, key=final_probs.get)
        return most_probable_final_state, final_probs[
            most_probable_final_state]


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
    parser.add_argument("--forward", type=str)

    args = parser.parse_args()

    # Create an HMM instance and load the model
    hmm = HMM()
    hmm.load(args.basename)

    # Generate and print a random sequence
    if args.generate:
        sequence = hmm.generate(args.generate)
        print(sequence)

    # Test the forward algorithm
    if args.forward:
        with open(args.forward, 'r') as f:
            observations = f.read().strip().split()
        most_probable_final_state, prob = hmm.forward(observations)
        print("\nForward Algorithm Results:")
        print("Most Probable Final State:", most_probable_final_state)
        print("Probability of Ending in This State:", prob)

        if args.basename == "lander":
            safe_states = ["2,5", "3,4", "4,3", "4,4", "5,5"]
            if most_probable_final_state in safe_states:
                print("Safe to land:", most_probable_final_state)
            else:
                print("Not safe to land:", most_probable_final_state)
