class MDP:
    """
    Represents a Markov Decision Process (MDP) model.
    This class defines the structure of an MDP, including states, actions,
    rewards, transition model, and discount factor. It can be initialized
    to a specific configuration, such as the Pittman Gridworld.
    """
    def __init__(self):
        """
        Initializes an empty MDP model.
        Specific MDP parameters (states, actions, etc.) are typically set by
        calling an initialization method like `initialize_pittman_gridworld_model`.
        """
        self.states = []  # List of all possible states, e.g., [(0,0), (0,1), ...]
        self.actions = [] # List of all possible actions, e.g., ["up", "down", ...]
        self.rewards = {} # Nested dictionary: self.rewards[state][action] = reward_value
                          # Represents R(s,a), the reward for taking action 'a' in state 's'.
        self.transition_model = {} # Nested dictionary: self.transition_model[state][action][next_state] = probability
                                   # Represents T(s,a,s'), the probability of transitioning to 'next_state'
                                   # from 'state' after taking 'action'.
        self.discount_factor = 0.0 # Gamma (γ), the discount factor for future rewards.
        self.terminal_states = [(0, 3), (1, 3)] # Predefined terminal states for Pittman Gridworld.
                                                # Note: These are hardcoded for this specific environment.
        self.displayable_algo_states = [] # States that are part of the algorithm's processing and display (e.g., not walls).
        self.algo_states = [] # States considered by algorithms (typically all states except walls).

    def initialize_pittman_gridworld_model(self, x_prob_value, reward_step_cost, discount_factor_from_gui):
        """
        Initializes the MDP model with parameters specific to the Pittman Gridworld environment.

        Args:
            x_prob_value (float): The percentage probability (0-100) of the agent moving in the intended direction.
                                  The remaining probability is split equally for side movements.
            reward_step_cost (float): The reward (cost) for taking a step in a non-terminal state.
                                      Typically a small negative value (e.g., -0.04).
            discount_factor_from_gui (float): The discount factor (gamma) to be used for future rewards.
        """
        # Define the grid states. (1,1) is a wall.
        self.states = [
            (0, 0), (0, 1), (0, 2), (0, 3), # Top row; (0,3) is a terminal state (+1)
            (1, 0), (1, 1), (1, 2), (1, 3), # Middle row; (1,1) is a wall, (1,3) is a terminal state (-1)
            (2, 0), (2, 1), (2, 2), (2, 3)  # Bottom row
        ]
        self.actions = ["up", "down", "left", "right"] # Standard actions in the gridworld.
        self.discount_factor = discount_factor_from_gui
        
        # Initialize rewards for all states and actions.
        self.rewards = {}
        for state in self.states:
            if state in self.terminal_states:
                # Actions taken *from* a terminal state yield no further reward, as the episode has ended.
                self.rewards[state] = {action: 0 for action in self.actions}
            elif state == (1, 1): # Wall state
                # Actions taken from the wall state (which isn't really possible as agent can't be in wall)
                # or leading into wall also result in zero reward from this structure.
                # The algorithms typically prevent movement into walls or handle rewards upon *hitting* a wall.
                self.rewards[state] = {action: 0 for action in self.actions}
            else:
                # Assign the standard step cost for all actions in non-terminal, non-wall states.
                self.rewards[state] = {action: reward_step_cost for action in self.actions}
        
        # Override rewards for actions taken *in* terminal states.
        # This is a specific convention for this application.
        # Standard MDP formulation often has terminal states having a value, and rewards are for transitions *into* them.
        # Here, it implies a reward is obtained for any action if one *were* in a terminal state.
        # The RL algorithms in mdp_rl_gridworld.py generally handle terminal state rewards/values correctly
        # upon transitioning *to* these states or by setting their V-values directly.
        self.rewards[(0, 3)] = {action: 1.0 for action in self.actions} # Positive terminal state
        self.rewards[(1, 3)] = {action: -1.0 for action in self.actions} # Negative terminal state
        
        # Calculate transition probabilities based on x_prob_value.
        prob_intended = x_prob_value / 100.0  # Probability of moving in the intended direction.
        prob_side = (100.0 - x_prob_value) / 200.0

        action_index_difference = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }
        side_actions = {
            "up": ["left", "right"], "down": ["left", "right"],
            "left": ["up", "down"], "right": ["up", "down"],
        }
        
        self.transition_model = {}
        grid_rows = 3
        grid_cols = 4
        
        for state in self.states:
            self.transition_model[state] = {}
            if state in self.terminal_states or state == (1, 1):
                for action in self.actions:
                    self.transition_model[state][action] = {state: 1.0}
                continue
            
            for action in self.actions:
                self.transition_model[state][action] = {}
                
                row, col = state
                row_diff, col_diff = action_index_difference[action]
                intended_next_state_candidate = (row + row_diff, col + col_diff)
                
                if not (0 <= intended_next_state_candidate[0] < grid_rows and \
                        0 <= intended_next_state_candidate[1] < grid_cols) or \
                        intended_next_state_candidate == (1, 1):
                    intended_next_state = state
                else:
                    intended_next_state = intended_next_state_candidate
                
                self.transition_model[state][action][intended_next_state] = prob_intended
                
                for side_action in side_actions[action]:
                    side_row_diff, side_col_diff = action_index_difference[side_action]
                    side_next_state_candidate = (row + side_row_diff, col + side_col_diff)
                    
                    if not (0 <= side_next_state_candidate[0] < grid_rows and \
                            0 <= side_next_state_candidate[1] < grid_cols) or \
                            side_next_state_candidate == (1, 1):
                        side_next_state = state
                    else:
                        side_next_state = side_next_state_candidate
                    self.transition_model[state][action][side_next_state] = \
                        self.transition_model[state][action].get(side_next_state, 0.0) + prob_side
        
        # Recalculate algo_states and displayable_algo_states after states are defined
        self.algo_states = [s for s in self.states if s != (1, 1)] # States used in algorithms (excluding the wall).
        # Displayable states are those that appear on the grid and are not walls.
        # This list is used by some UI functions for iterating over cells to display.
        self.displayable_algo_states = [(r,c) for r in range(3) for c in range(4) if (r,c) != (1,1)]
