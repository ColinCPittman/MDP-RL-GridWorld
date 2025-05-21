#!/usr/bin/env python
# coding: utf-8

# # Imports and Gloabls

# In[34]:


import tkinter as tk
import time
import random

# # Class Definitions

class GridworldApp:
    """
    Manages the main Tkinter application window, GUI components,
    and interactions for the Gridworld RL environment.
    """
    def __init__(self, master):
        """
        Initializes the GridworldApp.
        Args:
            master: The root Tkinter window.
        """
        self.master = master
        master.title("Gridworld Display")

        self.current_grid_mode = "v"  # or "q"
        self.ql_mode = False 
        self.cells = {} 
        self.current_algorithm_instance = None 

        self.grid_frame = tk.Frame(master)
        self.grid_frame.grid(row=0, column=0, sticky="nsew")

        self.control_panel_frame = tk.Frame(master)
        self.control_panel_frame.grid(row=1, column=0, sticky="ew")

        self._setup_control_panel()

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self._initialize_v_grid_display() 

    def _setup_control_panel(self):
        """Sets up the control panel with buttons and input fields."""
        # Algorithm execution buttons
        self.value_iteration_button = tk.Button(self.control_panel_frame, text="Run Value Iteration", command=self._run_value_iteration_clicked)
        self.value_iteration_button.grid(row=0, column=0, padx=5, pady=5)
        self.policy_iteration_button = tk.Button(self.control_panel_frame, text="Run Policy Iteration", command=self._run_policy_iteration_clicked)
        self.policy_iteration_button.grid(row=0, column=1, padx=5, pady=5)
        self.q_learning_button = tk.Button(self.control_panel_frame, text="Run Q-Learning", command=self._run_q_learning_clicked)
        self.q_learning_button.grid(row=0, column=2, padx=5, pady=5)
        self.epsilon_greedy_q_button = tk.Button(self.control_panel_frame, text="Run Epsilon Greedy", command=self._run_epsilon_greedy_clicked)
        self.epsilon_greedy_q_button.grid(row=0, column=3, padx=5, pady=5)
        self.reset_button = tk.Button(self.control_panel_frame, text="Run Decaying E-Greedy", command=lambda: self._run_epsilon_greedy_clicked(decaying=True))
        self.reset_button.grid(row=0, column=4, padx=5, pady=5)
        self.display_button = tk.Button(self.control_panel_frame, text="Cycle Display Mode", command=self._toggle_display_mode_clicked)
        self.display_button.grid(row=0, column=5, padx=5, pady=5)

        # Input fields for MDP parameters
        x_value_label = tk.Label(self.control_panel_frame, text="X Value (%):")
        x_value_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.x_value_entry = tk.Entry(self.control_panel_frame, width=5)
        self.x_value_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.x_value_entry.insert(0, "90")

        r_value_label = tk.Label(self.control_panel_frame, text="R Value (Reward):")
        r_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.r_value_entry = tk.Entry(self.control_panel_frame, width=5)
        self.r_value_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.r_value_entry.insert(0, "-0.04")

        a_value_label = tk.Label(self.control_panel_frame, text="Alpha (QL Rate):")
        a_value_label.grid(row=1, column=4, padx=5, pady=5, sticky="e")
        self.a_value_entry = tk.Entry(self.control_panel_frame, width=5)
        self.a_value_entry.grid(row=1, column=5, padx=5, pady=5, sticky="w")
        self.a_value_entry.insert(0, "0.5")
        
        epsilon_label = tk.Label(self.control_panel_frame, text="Epsilon (Explore %):")
        epsilon_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.epsilon_entry = tk.Entry(self.control_panel_frame, width=5)
        self.epsilon_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.epsilon_entry.insert(0, "0.001")

        discount_label = tk.Label(self.control_panel_frame, text="Discount Factor:")
        discount_label.grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.discount_entry = tk.Entry(self.control_panel_frame, width=5)
        self.discount_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.discount_entry.insert(0, "0.99")

        # Output and speed control
        self.output_label = tk.Label(self.control_panel_frame, text="", width=50, anchor="w")
        self.output_label.grid(row=3, column=4, columnspan=3, padx=5, pady=5, sticky="w") # Adjusted column span
        
        speed_slider_label = tk.Label(self.control_panel_frame, text="Speed Multiplier:")
        speed_slider_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.speed_slider = tk.Scale(self.control_panel_frame, from_=.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1) # Adjusted range for more control
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="ew")


    def _get_mdp_model_from_ui(self): 
        """
        Retrieves MDP parameters from UI entries and creates an MDP object.
        Performs input validation and shows error messages if necessary.
        Returns:
            MDP object if inputs are valid, None otherwise.
        """
        try:
            r_val = float(self.r_value_entry.get())
        except ValueError:
            self.set_status_message("Error: R Value (Reward) must be a number.")
            return None
        try:
            x_val = float(self.x_value_entry.get())
            if not (0 <= x_val <= 100):
                self.set_status_message("Error: X Value (intended move %) must be between 0 and 100.")
                return None
        except ValueError:
            self.set_status_message("Error: X Value must be a number.")
            return None
        try:
            disc_val = float(self.discount_entry.get())
            if not (0 <= disc_val <= 1):
                self.set_status_message("Error: Discount Factor must be between 0 and 1.")
                return None
        except ValueError:
            self.set_status_message("Error: Discount Factor must be a number.")
            return None
        
        # Validate Alpha and Epsilon here as they are used by algorithms
        try:
            float(self.a_value_entry.get()) # Just check if it's a float
        except ValueError:
            self.set_status_message("Error: Alpha(QL) must be a number.")
            return None
        try:
            float(self.epsilon_entry.get()) # Just check if it's a float
        except ValueError:
            self.set_status_message("Error: Epsilon (Explore %) must be a number.")
            return None


        states_list = [(r, c) for r in range(3) for c in range(4)]
        terminal_states_list = [(0, 3), (1, 3)]
        actions_list = ["up", "down", "left", "right"]
        rewards_dict = {}
        for state in states_list:
            if state in terminal_states_list: rewards_dict[state] = {action: 0 for action in actions_list}
            elif state == (1, 1): rewards_dict[state] = {action: 0 for action in actions_list} # Wall
            else: rewards_dict[state] = {action: r_val for action in actions_list}
        rewards_dict[(0, 3)] = {action: 1.0 for action in actions_list}
        rewards_dict[(1, 3)] = {action: -1.0 for action in actions_list}
        
        prob_intended = x_val / 100.0
        prob_side = (1.0 - prob_intended) / 2.0 
        
        action_index_difference = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        side_actions_map = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}
        transition_dict = {}
        grid_rows, grid_cols = 3, 4
        for state in states_list:
            transition_dict[state] = {}
            if state in terminal_states_list or state == (1, 1):
                for action in actions_list: transition_dict[state][action] = {state: 1.0}
                continue
            for action in actions_list:
                transition_dict[state][action] = {}
                row, col = state
                row_diff, col_diff = action_index_difference[action]
                intended_next_state_candidate = (row + row_diff, col + col_diff)
                if not (0 <= intended_next_state_candidate[0] < grid_rows and 0 <= intended_next_state_candidate[1] < grid_cols) or intended_next_state_candidate == (1, 1):
                    intended_next_state = state
                else: intended_next_state = intended_next_state_candidate
                
                transition_dict[state][action][intended_next_state] = transition_dict[state][action].get(intended_next_state, 0.0) + prob_intended
                
                for side_action in side_actions_map[action]:
                    side_row_diff, side_col_diff = action_index_difference[side_action]
                    side_next_state_candidate = (row + side_row_diff, col + side_col_diff)
                    if not (0 <= side_next_state_candidate[0] < grid_rows and 0 <= side_next_state_candidate[1] < grid_cols) or side_next_state_candidate == (1, 1):
                        side_next_state = state
                    else: side_next_state = side_next_state_candidate
                    transition_dict[state][action][side_next_state] = transition_dict[state][action].get(side_next_state, 0.0) + prob_side
        return MDP(states_list, actions_list, rewards_dict, transition_dict, disc_val)

    def _initialize_q_grid_display(self): 
        """Initializes or clears the grid for Q-value display."""
        self.set_status_message("")
        num_displayable_cells = 9 
        initial_q_quadtuples = [("0.00", "0.00", "0.00", "0.00")] * num_displayable_cells
        self._draw_q_grid_elements(initial_q_quadtuples, q_learn_active=self.ql_mode)

    def _initialize_v_grid_display(self): 
        """Initializes or clears the grid for V-value/Policy display."""
        self.set_status_message("")
        self.current_grid_mode = "v"
        num_displayable_cells = 9 
        initial_v_tuples = [("0.00", "up")] * num_displayable_cells
        self._draw_v_grid_elements(initial_v_tuples) 

    def _initialize_grid_display(self): 
        """Calls the appropriate grid initialization based on the current mode."""
        if self.current_grid_mode == "v": self._initialize_v_grid_display()
        else: self._initialize_q_grid_display()

    def refresh_display_from_algorithm_data(self, display_data, iteration_delay_info=None):
        """
        Refreshes the grid display based on data from an algorithm.
        Args:
            display_data (dict): Contains data for display. Expected keys:
                                 'v_display_tuples' (for V-mode),
                                 'q_display_quads' (for Q-mode),
                                 'current_agent_state', 'q_learn_active'.
            iteration_delay_info (tuple, optional): (should_delay_bool, delay_factor_float).
                                                    Controls whether to use `perform_iteration_delay`.
        """
        v_display_tuples = display_data.get('v_display_tuples')
        q_display_quads = display_data.get('q_display_quads')
        current_agent_state = display_data.get('current_agent_state')
        q_learn_active_flag = display_data.get('q_learn_active', self.ql_mode)

        if self.current_grid_mode == "v":
            if v_display_tuples is not None:
                 self._draw_v_grid_elements(v_display_tuples, type=display_data.get('display_type'))
            elif q_display_quads is not None: # Fallback for algorithms that compute Q to show in V-mode
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        else: # Q-mode display
            if q_display_quads is not None:
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        
        if iteration_delay_info and iteration_delay_info[0]: 
            self.perform_iteration_delay(iteration_delay_info[1])
        else: 
             self.master.update_idletasks()

    def _toggle_display_mode_clicked(self): 
        """Cycles between V-value/Policy display and Q-value display."""
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.cells.clear() 
        if self.current_grid_mode == "v":
            self.current_grid_mode = "q"; self._initialize_q_grid_display()
        else:
            self.current_grid_mode = "v"; self._initialize_v_grid_display()
        
        # If an algorithm has run and has data, refresh display in new mode
        if self.current_algorithm_instance and hasattr(self.current_algorithm_instance, 'get_current_display_data_for_toggle'):
            data = self.current_algorithm_instance.get_current_display_data_for_toggle()
            self.refresh_display_from_algorithm_data(data, iteration_delay_info=(False, 0))

    def _draw_v_grid_elements(self, v_policy_tuples_list, type=None): 
        """Draws the V-values and policy arrows (or just policy)."""
        if len(v_policy_tuples_list) != 9: return 
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.cells.clear() 
        for i in range(3): self.grid_frame.grid_rowconfigure(i, weight=1, minsize=100)
        for j in range(4): self.grid_frame.grid_columnconfigure(j, weight=1, minsize=100)
        
        tuple_index = 0
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                is_terminal_pos, is_terminal_neg, is_wall_cell = cell_pos == (0,3), cell_pos == (1,3), cell_pos == (1,1)
                current_text = ""
                if is_terminal_pos: current_text = "1.00"
                elif is_terminal_neg: current_text = "-1.00"
                elif is_wall_cell: current_text = ""
                else: # Regular, non-terminal, non-wall cells
                    if tuple_index < len(v_policy_tuples_list):
                        v_score_str, direction_str = v_policy_tuples_list[tuple_index]
                        current_text = f"Max Reward:\n\n{v_score_str} if going {direction_str}." if type is None else direction_str
                        tuple_index += 1
                    else: current_text = "N/A" # Should not happen if list is correct

                cell_label = tk.Label(self.grid_frame, text=current_text, relief=tk.SOLID, padx=10, pady=5, width=10, height=5, font=("Comfortaa", 12))
                cell_label.grid(row=row_idx, column=col_idx, sticky="nsew")
                if is_wall_cell: cell_label.config(bg="grey")
                self.cells[cell_pos] = cell_label
                
    def _draw_q_grid_elements(self, quad_tuples_list, q_learn_active=False, current_agent_state=None):
        """Draws the Q-values in their respective directional slots within each cell."""
        if len(quad_tuples_list) != 9: return
        quad_idx = 0
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                is_terminal_pos, is_terminal_neg, is_wall_cell = cell_pos == (0,3), cell_pos == (1,3), cell_pos == (1,1)

                if cell_pos not in self.cells or not isinstance(self.cells[cell_pos], dict):
                    cell_frame = tk.Frame(self.grid_frame, relief=tk.SOLID, bd=1, width=100, height=100)
                    cell_frame.grid(row=row_idx, column=col_idx, sticky="nsew")
                    cell_frame.grid_propagate(False)
                    for r_s in range(3): cell_frame.grid_rowconfigure(r_s, weight=1)
                    for c_s in range(3): cell_frame.grid_columnconfigure(c_s, weight=1)
                    
                    self.cells[cell_pos] = { 'frame': cell_frame,
                        'center': tk.Label(cell_frame, text="", font=("Comfortaa", 10)),
                        'top': tk.Label(cell_frame, text="0.00", anchor="s"), 
                        'right': tk.Label(cell_frame, text="0.00", anchor="w"),
                        'bottom': tk.Label(cell_frame, text="0.00", anchor="n"), 
                        'left': tk.Label(cell_frame, text="0.00", anchor="e")}
                    
                    self.cells[cell_pos]['center'].grid(row=1, column=1)
                    self.cells[cell_pos]['top'].grid(row=0, column=1, sticky="ew")
                    self.cells[cell_pos]['right'].grid(row=1, column=2, sticky="ns")
                    self.cells[cell_pos]['bottom'].grid(row=2, column=1, sticky="ew")
                    self.cells[cell_pos]['left'].grid(row=1, column=0, sticky="ns")

                cell_ui = self.cells[cell_pos]
                center_text_val = ""
                if current_agent_state == cell_pos and self.current_algorithm_instance and \
                   cell_pos not in self.current_algorithm_instance.mdp.terminal_states and not is_wall_cell:
                    center_text_val = "here"
                
                if is_terminal_pos: 
                    cell_ui['center'].config(text="1.00" if not q_learn_active else center_text_val, font=("Comfortaa", 12))
                    for d in ['top','right','bottom','left']: cell_ui[d].config(text="")
                elif is_terminal_neg: 
                    cell_ui['center'].config(text="-1.00" if not q_learn_active else center_text_val, font=("Comfortaa", 12))
                    for d in ['top','right','bottom','left']: cell_ui[d].config(text="")
                elif is_wall_cell: 
                    cell_ui['frame'].config(bg="grey")
                    for lbl in ['center','top','right','bottom','left']: cell_ui[lbl].config(text="", bg="grey")
                else: # Regular, non-terminal, non-wall cells
                    if quad_idx < len(quad_tuples_list):
                        quad_data = quad_tuples_list[quad_idx]
                        cell_ui['top'].config(text=f"{quad_data[0]}")
                        cell_ui['right'].config(text=f"{quad_data[1]}")
                        cell_ui['bottom'].config(text=f"{quad_data[2]}")
                        cell_ui['left'].config(text=f"{quad_data[3]}")
                        cell_ui['center'].config(text=center_text_val)
                        quad_idx +=1

    def set_status_message(self, message): 
        """Updates the status message label in the GUI."""
        self.output_label.config(text=message)

    def perform_iteration_delay(self, base_delay=0.2):
        """Handles GUI updates and introduces a delay for visualization."""
        if base_delay and base_delay > 0: 
            self.master.update() # Process all pending Tkinter events
            time.sleep(base_delay / float(self.speed_slider.get()))
        else: 
            self.master.update_idletasks() # Process only idle tasks (less disruptive)

    def _run_generic_algorithm(self, algorithm_class, **kwargs):
        """Helper to stop current algorithm and start a new one."""
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
            self.current_algorithm_instance.stop()
        
        mdp_model = self._get_mdp_model_from_ui()
        if mdp_model is None:  # Input validation failed
            return
        
        self.current_algorithm_instance = algorithm_class(mdp_model, self, **kwargs)
        if hasattr(self.current_algorithm_instance, 'start') and callable(getattr(self.current_algorithm_instance, 'start')):
            self.current_algorithm_instance.start()
        elif hasattr(self.current_algorithm_instance, 'run_algorithm') and callable(getattr(self.current_algorithm_instance, 'run_algorithm')):
             # For QLearning which has a blocking loop
            self.current_algorithm_instance.run_algorithm()


    def _run_value_iteration_clicked(self):
        self._run_generic_algorithm(ValueIterationAlgorithm)
    def _run_policy_iteration_clicked(self):
        self._run_generic_algorithm(PolicyIterationAlgorithm)
    def _run_q_learning_clicked(self):
        self._run_generic_algorithm(QLearningAlgorithm)
    def _run_epsilon_greedy_clicked(self, decaying=False):
        self._run_generic_algorithm(EpsilonGreedyAlgorithm, decaying_epsilon=decaying)


class MDP:
    """
    Represents a Markov Decision Process, defining the environment.
    """
    def __init__(self, states, actions, rewards, transition_model, discount_factor):
        self.states, self.actions, self.rewards, self.transition_model, self.discount_factor = \
            states, actions, rewards, transition_model, discount_factor
        self.terminal_states = [(0, 3), (1, 3)] 
        # 9 cells that are not walls or terminals (used for V/Q iteration, policy)
        self.displayable_algo_states = [(r,c) for r in range(3) for c in range(4) if (r,c) != (1,1)] 
        # All non-wall states (11 states, including terminals)
        self.algo_states = [s for s in self.states if s != (1,1)] 

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """
    def __init__(self, mdp_model, app_interface): 
        self.mdp, self.app = mdp_model, app_interface 
        self.V = {s: 0.0 for s in self.mdp.algo_states} # Value function
        self.Q = {s: {a: 0.0 for a in self.mdp.actions} for s in self.mdp.algo_states} # Q-value function
        self.policy = {s: None for s in self.mdp.algo_states} # Policy
        self.is_running = False # Flag to control asynchronous execution
        self.iteration = 0      # Current iteration count
        self.max_iterations = 2000 # Default safeguard for non-converging loops

    def start(self): 
        """Starts the algorithm's execution (typically using Tkinter's after)."""
        self.is_running = True
        self.iteration = 0
        raise NotImplementedError("Subclasses must implement the start method.")
        
    def stop(self):
        """Stops the algorithm's execution."""
        self.is_running = False

    def _prepare_v_display_tuples(self):
        """Formats V-values and policy into a flat list for GUI display."""
        tuples = []
        # Order matters for direct mapping to grid cells in _draw_v_grid_elements
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        for state_coord in ordered_display_cells:
            tuples.append((f"{self.V.get(state_coord, 0.0):.2f}", self.policy.get(state_coord, "")))
        return tuples
    
    def _prepare_q_display_quads(self, q_data_source=None):
        """Formats Q-values into a flat list of string tuples for GUI display."""
        quads = []
        source = q_data_source if q_data_source is not None else self.Q
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        for state_coord in ordered_display_cells:
            q_s_vals = source.get(state_coord, {}) # Default to empty if state not in Q dict
            quads.append((f"{q_s_vals.get('up', 0.0):.2f}", f"{q_s_vals.get('right', 0.0):.2f}",
                          f"{q_s_vals.get('down', 0.0):.2f}", f"{q_s_vals.get('left', 0.0):.2f}"))
        return quads

    def get_current_display_data_for_toggle(self):
        """
        Prepares data suitable for display when toggling between V-mode and Q-mode.
        This often involves deriving Q-values from V for VI/PI, or policy from Q for QL/EG.
        """
        v_tuples = self._prepare_v_display_tuples()
        q_for_display_dict = {} # This will hold Q(s,a) for all actions
        
        if isinstance(self, (ValueIterationAlgorithm, PolicyIterationAlgorithm)):
            # For VI/PI, Q-values for display are derived from the current V-function
            for s in self.mdp.algo_states: 
                if s in self.mdp.terminal_states: 
                    q_for_display_dict[s] = {a: self.V.get(s,0.0) for a in self.mdp.actions} # Terminal Q is its V
                    continue
                action_qs = {}
                for a in self.mdp.actions:
                    expected_val = 0
                    if s in self.mdp.transition_model and a in self.mdp.transition_model[s]:
                        for next_s, prob in self.mdp.transition_model[s][a].items():
                            if next_s != (1,1): expected_val += prob * self.V.get(next_s,0.0)
                    action_qs[a] = self.mdp.rewards[s][a] + self.mdp.discount_factor * expected_val
                q_for_display_dict[s] = action_qs
            q_display_quads = self._prepare_q_display_quads(q_data_source=q_for_display_dict)
        else: # QLearning or EpsilonGreedy, self.Q is the direct source
            q_display_quads = self._prepare_q_display_quads()
            # Derive policy from self.Q for V-mode display
            for s_pol in self.mdp.algo_states: 
                if s_pol not in self.mdp.terminal_states and self.Q.get(s_pol):
                    self.policy[s_pol] = max(self.Q[s_pol], key=self.Q[s_pol].get)


        return {'v_display_tuples': v_tuples, 'q_display_quads': q_display_quads, 
                'current_agent_state': getattr(self, 'current_state', None), 
                'q_learn_active': isinstance(self, (QLearningAlgorithm, EpsilonGreedyAlgorithm))}

class ValueIterationAlgorithm(RLAlgorithm):
    """Implements the Value Iteration algorithm."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        # Initialize terminal state values
        if (0,3) in self.V: self.V[(0,3)] = self.mdp.rewards.get((0,3),{}).get("up", 1.0) 
        if (1,3) in self.V: self.V[(1,3)] = self.mdp.rewards.get((1,3),{}).get("up", -1.0)

    def start(self):
        """Starts the Value Iteration process."""
        self.is_running = True; self.iteration = 0
        self.app._initialize_grid_display(); self.app.ql_mode = False 
        self._perform_one_iteration()

    def _perform_one_iteration(self):
        """Performs a single iteration of Value Iteration."""
        if not self.is_running: return
        
        try: epsilon = float(self.app.epsilon_entry.get())
        except ValueError: self.app.set_status_message("Error: Epsilon must be a number."); self.is_running = False; return
        
        threshold = epsilon * (1 - self.mdp.discount_factor) / self.mdp.discount_factor if self.mdp.discount_factor != 1.0 else epsilon
        
        delta = 0; new_V = self.V.copy(); q_values_for_v_iteration = {}
        for s in self.mdp.algo_states:
            if s in self.mdp.terminal_states: continue 
            max_action_value = float('-inf'); current_action_q_values = {}
            for a in self.mdp.actions:
                action_reward = self.mdp.rewards[s][a]; expected_future_value = 0
                if s in self.mdp.transition_model and a in self.mdp.transition_model[s]:
                    for next_s, prob in self.mdp.transition_model[s][a].items():
                        if next_s != (1,1): expected_future_value += prob * self.V[next_s]
                current_action_q_values[a] = action_reward + self.mdp.discount_factor * expected_future_value
                max_action_value = max(max_action_value, current_action_q_values[a])
            new_V[s] = max_action_value
            q_values_for_v_iteration[s] = current_action_q_values
            delta = max(delta, abs(new_V[s] - self.V[s]))
        self.V = new_V

        for s_policy in self.mdp.algo_states:
            if s_policy not in self.mdp.terminal_states and q_values_for_v_iteration.get(s_policy):
                self.policy[s_policy] = max(q_values_for_v_iteration[s_policy], key=q_values_for_v_iteration[s_policy].get)
        
        v_tuples = self._prepare_v_display_tuples()
        q_quads = self._prepare_q_display_quads(q_data_source=q_values_for_v_iteration)
        
        self.app.refresh_display_from_algorithm_data({
            'v_display_tuples': v_tuples, 'q_display_quads': q_quads, 'q_learn_active': False
        }, iteration_delay_info=(False, 0)) 
        
        self.iteration +=1
        if delta > threshold and self.iteration < self.max_iterations:
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000) 
            self.app.master.after(delay_ms, self._perform_one_iteration)
        else:
            status = 'converged' if delta <= threshold else 'stopped (max iterations)'
            self.app.set_status_message(f"Value iteration {status} after {self.iteration} iterations.")
            self.is_running = False

class PolicyIterationAlgorithm(RLAlgorithm):
    """Implements the Policy Iteration algorithm."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        if (0,3) in self.V: self.V[(0,3)] = self.mdp.rewards.get((0,3),{}).get("up",1.0)
        if (1,3) in self.V: self.V[(1,3)] = self.mdp.rewards.get((1,3),{}).get("up",-1.0)
        for s_init in self.mdp.algo_states:
            if s_init not in self.mdp.terminal_states: self.policy[s_init] = random.choice(self.mdp.actions) 

    def start(self):
        """Starts the Policy Iteration process."""
        self.is_running = True; self.iteration = 0
        self.app._initialize_grid_display(); self.app.ql_mode = False
        self._perform_one_policy_iteration_step()

    def _policy_evaluation(self, epsilon_eval): 
        """Performs the policy evaluation step (iterative)."""
        threshold_eval = epsilon_eval * (1 - self.mdp.discount_factor) / self.mdp.discount_factor if self.mdp.discount_factor != 1.0 else epsilon_eval
        eval_V = self.V.copy() # Start with current V values
        while True: 
            delta_eval = 0; new_eval_V = eval_V.copy()
            for s_eval in self.mdp.algo_states:
                if s_eval in self.mdp.terminal_states: continue
                old_v_s_eval = eval_V[s_eval]; action_to_eval = self.policy[s_eval]
                if action_to_eval is None: new_eval_V[s_eval] = 0; continue # Should not happen with init
                
                action_reward_eval = self.mdp.rewards[s_eval][action_to_eval]; expected_future_val_eval = 0
                if s_eval in self.mdp.transition_model and action_to_eval in self.mdp.transition_model[s_eval]:
                     for next_s_eval, prob_eval in self.mdp.transition_model[s_eval][action_to_eval].items():
                        if next_s_eval != (1,1): expected_future_val_eval += prob_eval * eval_V[next_s_eval]
                new_eval_V[s_eval] = action_reward_eval + self.mdp.discount_factor * expected_future_val_eval
                delta_eval = max(delta_eval, abs(new_eval_V[s_eval] - old_v_s_eval))
            eval_V = new_eval_V
            if delta_eval <= threshold_eval: break
        return eval_V

    def _perform_one_policy_iteration_step(self):
        """Performs one step of policy iteration (evaluation + improvement)."""
        if not self.is_running: return
        try: epsilon = float(self.app.epsilon_entry.get())
        except ValueError: self.app.set_status_message("Error: Epsilon must be a number."); self.is_running = False; return

        self.V = self._policy_evaluation(epsilon) 
        policy_stable = True; q_for_improvement = {} 
        for s_improve in self.mdp.algo_states:
            if s_improve in self.mdp.terminal_states: 
                q_for_improvement[s_improve] = {a:self.V.get(s_improve,0.0) for a in self.mdp.actions}
                continue
            old_action_improve = self.policy[s_improve]; action_q_s_improve = {}
            for a_improve in self.mdp.actions:
                action_reward_improve = self.mdp.rewards[s_improve][a_improve]; expected_future_val_improve = 0
                if s_improve in self.mdp.transition_model and a_improve in self.mdp.transition_model[s_improve]:
                    for next_s_improve, prob_improve in self.mdp.transition_model[s_improve][a_improve].items():
                        if next_s_improve != (1,1): expected_future_val_improve += prob_improve * self.V[next_s_improve]
                action_q_s_improve[a_improve] = action_reward_improve + self.mdp.discount_factor * expected_future_val_improve
            q_for_improvement[s_improve] = action_q_s_improve
            if action_q_s_improve:
                best_action_improve = max(action_q_s_improve, key=action_q_s_improve.get)
                self.policy[s_improve] = best_action_improve
                if best_action_improve != old_action_improve: policy_stable = False
        
        v_tuples_pi, q_quads_pi = self._prepare_flat_display_data_from_v_and_q_dict(q_for_improvement)
        self.app.refresh_display_from_algorithm_data({
            'v_display_tuples': v_tuples_pi, 'q_display_quads': q_quads_pi, 'q_learn_active': False
        }, iteration_delay_info=(False, 0))
        
        self.iteration +=1
        if not policy_stable and self.iteration < self.max_iterations :
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000)
            self.app.master.after(delay_ms, self._perform_one_policy_iteration_step)
        else:
            status = 'converged' if policy_stable else 'stopped (max iterations)'
            self.app.set_status_message(f"Policy iteration {status} after {self.iteration} iterations.")
            self.is_running = False
    
    def _prepare_flat_display_data_from_v_and_q_dict(self, q_values_dict):
        v_display_tuples = []; q_display_quads = []
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        for state_coord in ordered_display_cells:
            v_score_str = f"{self.V.get(state_coord, 0.0):.2f}"; direction_str = self.policy.get(state_coord, "")
            v_display_tuples.append((v_score_str, direction_str))
            q_s_vals = q_values_dict.get(state_coord, {})
            q_display_quads.append(( f"{q_s_vals.get('up', 0.0):.2f}", f"{q_s_vals.get('right', 0.0):.2f}",
                                     f"{q_s_vals.get('down', 0.0):.2f}", f"{q_s_vals.get('left', 0.0):.2f}" ))
        return v_display_tuples, q_display_quads

class QLearningAlgorithm(RLAlgorithm):
    """Implements the Q-Learning algorithm (user-interactive version)."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        self.N_sa = {s: {a: 0 for a in self.mdp.actions} for s in self.mdp.algo_states}
        self.current_state = (2,0) 
        self.move_var = tk.StringVar()
        for terminal_s in self.mdp.terminal_states: 
            if terminal_s in self.Q:
                for a_term in self.mdp.actions: self.Q[terminal_s][a_term] = 0.0 # Q-values at terminal are 0

    def _on_key(self, event):
        if event.keysym in ['Up', 'Down', 'Left', 'Right']: self.move_var.set(event.keysym.lower())

    def run_algorithm(self): 
        """Runs the Q-learning algorithm, driven by user key presses."""
        self.is_running = True 
        self.app.ql_mode = True; self.app.current_grid_mode = 'q'; self.app._initialize_q_grid_display()
        try: initial_alpha = float(self.app.a_value_entry.get())
        except ValueError: self.app.set_status_message("Error: Alpha(QL) must be a number."); self.is_running = False; return
        
        move_count = 0; max_moves = 40 # Limit manual moves
        self.app.master.bind('<Key>', self._on_key)
        self.app.set_status_message("Use the arrow keys to move the marker.")
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))

        while move_count < max_moves:
            self.app.master.wait_variable(self.move_var); action = self.move_var.get(); self.move_var.set("")
            if not self.is_running: break # Allow stopping if new algo starts
            if not action or action not in self.mdp.actions : continue
            
            s_next = self.current_state 
            if self.current_state in self.mdp.transition_model and action in self.mdp.transition_model[self.current_state]:
                probs = self.mdp.transition_model[self.current_state][action]
                next_states_list, probabilities_list = list(probs.keys()), list(probs.values())
                if next_states_list: s_next = random.choices(next_states_list, weights=probabilities_list, k=1)[0]
            
            r_val = self.mdp.rewards[self.current_state][action] 
            if s_next == (0,3): r_val = 1.0
            elif s_next == (1,3): r_val = -1.0
            
            if self.current_state not in self.mdp.terminal_states:
                max_next_q_val = 0.0
                if s_next in self.Q and s_next not in self.mdp.terminal_states: 
                    max_next_q_val = max(self.Q[s_next].values()) if self.Q[s_next] else 0.0
                
                target_q_val = r_val + self.mdp.discount_factor * max_next_q_val
                self.N_sa[self.current_state][action] += 1 
                alpha = initial_alpha / (1 + self.N_sa[self.current_state][action])
                self.Q[self.current_state][action] = (1-alpha)*self.Q[self.current_state][action] + alpha * target_q_val
            
            self.current_state = s_next
            self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(True, 0.1)) 
            move_count += 1
            
            if self.current_state in self.mdp.terminal_states:
                self.app.perform_iteration_delay(0.5); self.current_state = (2,0) 
                self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        
        self.app.set_status_message(f"Q-Learning stopped after {move_count} moves."); self.app.master.unbind('<Key>')
        self.is_running = False

class EpsilonGreedyAlgorithm(QLearningAlgorithm): 
    """Implements Epsilon-Greedy Q-Learning (automated steps)."""
    def __init__(self, mdp_model, app_interface, decaying_epsilon=False):
        super().__init__(mdp_model, app_interface); self.decaying_epsilon = decaying_epsilon
        self.max_moves = 300 
        self.move_count = 0 

    def start(self):
        """Starts the Epsilon-Greedy Q-Learning process."""
        self.is_running = True; self.move_count = 0
        self.app.ql_mode = True; self.app.current_grid_mode = 'q'; self.app._initialize_q_grid_display() 
        self.app.set_status_message("Running Epsilon-Greedy...")
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        self._perform_one_eg_step()

    def _perform_one_eg_step(self):
        """Performs a single step of Epsilon-Greedy Q-Learning."""
        if not self.is_running or self.move_count >= self.max_moves:
            if self.is_running : self.app.set_status_message(f"Epsilon-Greedy finished after {self.move_count} moves.")
            self.is_running = False; return

        try: 
            initial_alpha = float(self.app.a_value_entry.get())
            epsilon_val = float(self.app.epsilon_entry.get())
        except ValueError:
            self.app.set_status_message("Error: Alpha or Epsilon is not a valid number.")
            self.is_running = False; return

        action_chosen = random.choice(self.mdp.actions) if random.random() < epsilon_val else \
                        max(self.Q[self.current_state], key=self.Q[self.current_state].get) if self.current_state in self.Q and self.Q[self.current_state] else random.choice(self.mdp.actions)
        
        s_next = self.current_state 
        if self.current_state in self.mdp.transition_model and action_chosen in self.mdp.transition_model[self.current_state]:
            probs_eg = self.mdp.transition_model[self.current_state][action_chosen]
            next_states_list_eg, probabilities_list_eg = list(probs_eg.keys()), list(probs_eg.values())
            if next_states_list_eg: s_next = random.choices(next_states_list_eg, weights=probabilities_list_eg, k=1)[0]
        
        r_val_eg = self.mdp.rewards[self.current_state][action_chosen]
        if s_next == (0,3): r_val_eg = 1.0
        elif s_next == (1,3): r_val_eg = -1.0
        
        if self.current_state not in self.mdp.terminal_states:
            max_next_q_val_eg = 0.0
            if s_next in self.Q and s_next not in self.mdp.terminal_states:
                max_next_q_val_eg = max(self.Q[s_next].values()) if self.Q[s_next] else 0.0
            target_q_val_eg = r_val_eg + self.mdp.discount_factor * max_next_q_val_eg
            self.N_sa[self.current_state][action_chosen] += 1 
            alpha_eg = initial_alpha / (1 + self.N_sa[self.current_state][action_chosen])
            self.Q[self.current_state][action_chosen] = (1-alpha_eg)*self.Q[self.current_state][action_chosen] + alpha_eg * target_q_val_eg
        
        self.current_state = s_next
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        self.move_count += 1
        
        if self.decaying_epsilon:
            epsilon_val = max(0.001, epsilon_val * 0.99)  
            self.app.epsilon_entry.delete(0, tk.END); self.app.epsilon_entry.insert(0, str(f"{epsilon_val:.4f}"))
        
        delay_ms_eg = int((0.05 / float(self.app.speed_slider.get())) * 1000)
        next_step_delay_after = delay_ms_eg
        
        if self.current_state in self.mdp.terminal_states:
            self.current_state = (2,0) 
            self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0)) 
            next_step_delay_after = 500 # Pause after reset from terminal
            
        self.app.master.after(next_step_delay_after, self._perform_one_eg_step)
    
# # Main Controller
# In[ ]:

def main():
    root = tk.Tk()
    app = GridworldApp(root) 
    root.mainloop() 

if __name__ == "__main__":
    main()

[end of src/mdp_rl_gridworld.py]
