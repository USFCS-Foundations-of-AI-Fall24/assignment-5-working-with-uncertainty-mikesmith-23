import random
import argparse
import numpy


class Sequence:
    """
    Represents a sequence of hidden states and corresponding output variables.
    """

    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # Sequence of states
        self.outputseq = outputseq  # Sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


class HMM:
    """
    Hidden Markov Model class that includes methods for loading model parameters,
    generating sequences, and performing Forward and Viterbi algorithms.
    """

    def __init__(self, transitions=None, emissions=None):
        self.transitions = transitions or {}  # Transition probabilities
        self.emissions = emissions or {}  # Emission probabilities

    def load(self, basename):
        """
        Reads HMM structure from transition and emission files,
        storing probabilities in self.transitions and self.emissions.
        """
        self.transitions = {}
        # Load transitions
        with open(f"{basename}.trans", 'r') as trans_file:
            for line in trans_file:
                parts = line.strip().split()
                if len(parts) == 3:
                    source_state, target_state, prob = parts
                    self.transitions.setdefault(source_state, {})[target_state] = float(prob)

        self.emissions = {}
        # Load emissions
        with open(f"{basename}.emit", 'r') as emit_file:
            for line in emit_file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, observation, prob = parts
                    self.emissions.setdefault(state, {})[observation] = float(prob)

    def generate(self, n):
        """
        Generates an n-length sequence of states and observations
        by randomly sampling from the HMM.
        """
        current_state = "#"  # Start at the initial state
        states, observations = [], []

        for _ in range(n):
            # Sample next state
            next_states = list(self.transitions[current_state].keys())
            next_probs = list(self.transitions[current_state].values())
            current_state = random.choices(next_states, weights=next_probs)[0]
            states.append(current_state)

            # Stop if a terminal state is reached
            if current_state == "5,5":
                break

            # Sample observation
            obs_choices = list(self.emissions[current_state].keys())
            obs_probs = list(self.emissions[current_state].values())
            observations.append(random.choices(obs_choices, weights=obs_probs)[0])

        return Sequence(states, observations)

    def initialize_probabilities(self, observations, prob_matrix, backtrack_matrix=None):
        """
        Initializes the probabilities for the first step of the algorithms
        (used by both Forward and Viterbi).
        """
        hidden_states = [state for state in self.transitions.keys() if state != "#"]

        for state in hidden_states:
            if "#" in self.transitions and state in self.transitions["#"]:
                init_prob = self.transitions["#"][state]
                emit_prob = self.emissions[state].get(observations[0], 0)
                prob_matrix[state][0] = init_prob * emit_prob
                if backtrack_matrix is not None:
                    backtrack_matrix[state][0] = None

        return hidden_states

    def forward(self, observations):
        """
        Implements the Forward algorithm to compute the most probable
        final state for a sequence of observations.
        """
        num_obs = len(observations)
        forward_matrix = {state: [0] * num_obs for state in self.transitions if state != "#"}
        hidden_states = self.initialize_probabilities(observations, forward_matrix)

        # Compute probabilities for each step
        for obs_index in range(1, num_obs):
            for current_state in hidden_states:
                total_prob = sum(
                    forward_matrix[prev_state][obs_index - 1] *
                    self.transitions[prev_state].get(current_state, 0)
                    for prev_state in hidden_states
                )
                forward_matrix[current_state][obs_index] = total_prob * self.emissions[current_state].get(
                    observations[obs_index], 0
                )

        # Find the most probable final state
        final_probs = {state: forward_matrix[state][num_obs - 1] for state in hidden_states}
        most_probable_final_state = max(final_probs, key=final_probs.get)
        return most_probable_final_state, final_probs[most_probable_final_state]

    def viterbi(self, observations):
        """
        Implements the Viterbi algorithm to compute the most likely
        sequence of states for a given sequence of observations.
        """
        num_obs = len(observations)
        prob_matrix = {state: [0] * num_obs for state in self.transitions if state != "#"}
        backtrack_matrix = {state: [None] * num_obs for state in prob_matrix}
        hidden_states = self.initialize_probabilities(observations, prob_matrix, backtrack_matrix)

        # Compute probabilities for each step
        for obs_index in range(1, num_obs):
            for curr_state in hidden_states:
                max_prob, best_prev_state = max(
                    (prob_matrix[prev_state][obs_index - 1] *
                     self.transitions[prev_state].get(curr_state, 0), prev_state)
                    for prev_state in hidden_states
                )
                prob_matrix[curr_state][obs_index] = max_prob * self.emissions[curr_state].get(
                    observations[obs_index], 0
                )
                backtrack_matrix[curr_state][obs_index] = best_prev_state

        # Find the most likely path
        max_final_prob, best_final_state = max(
            (prob_matrix[state][num_obs - 1], state) for state in hidden_states
        )
        most_likely_path = [best_final_state]
        for obs_index in range(num_obs - 1, 0, -1):
            most_likely_path.insert(0, backtrack_matrix[most_likely_path[0]][obs_index])

        return Sequence(most_likely_path, observations)


if __name__ == "__main__":

    # python3 HMM.py cat --generate 20
    # python3 HMM.py lander --generate 20
    # python3 HMM.py partofspeech --generate 20
    # python3 HMM.py cat --forward cat_sequence.obs
    # python3 HMM.py lander --forward lander_sequence.obs
    # python3 HMM.py cat --viterbi cat_sequence.obs
    # python3 HMM.py partofspeech --viterbi ambiguous_sents.obs
    
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", type=str)
    parser.add_argument("--generate", type=int)
    parser.add_argument("--forward", type=str)
    parser.add_argument("--viterbi", type=str)
    args = parser.parse_args()

    # Load the HMM model
    hmm = HMM()
    hmm.load(args.basename)

    # Generate and print random sequence
    if args.generate:
        sequence = hmm.generate(args.generate)
        print("Generated Sequence of States:", sequence.stateseq)
        print("Generated Sequence of Observations:", sequence.outputseq)

    # Test the Forward algorithm
    if args.forward:
        with open(args.forward, 'r') as f:
            observations = f.read().strip().split()
        most_probable_final_state, prob = hmm.forward(observations)
        print("\nForward Algorithm Results:")
        print("Most Probable Final State:", most_probable_final_state)
        print("Probability of Ending in This State:", prob)

    # Test the Viterbi algorithm
    if args.viterbi:
        with open(args.viterbi, 'r') as f:
            observations = f.read().strip().split()
        sequence = hmm.viterbi(observations)
        print("Viterbi Sequence of States:", sequence.stateseq)
        print("Viterbi Observations:", sequence.outputseq)
