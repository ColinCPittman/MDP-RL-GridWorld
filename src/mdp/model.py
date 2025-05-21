class MDP:
    """Represents a Markov Decision Process (MDP) model."""
    def __init__(self, states, actions, rewards, transition_model, discount_factor):
        self.states = states
        self.actions = actions
        self.rewards = rewards                  # R(s,a) : reward for action a in state s
        self.transition_model = transition_model # T(s,a,s') : probability of s' from (s,a)
        self.discount_factor = discount_factor
        self.terminal_states = [(0, 3), (1, 3)] 
        self.displayable_algo_states = [(r,c) for r in range(3) for c in range(4) if (r,c) != (1,1)] 
        self.algo_states = [s for s in self.states if s != (1,1)] # Excludes wall
