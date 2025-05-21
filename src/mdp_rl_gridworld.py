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

        # --- UI Configuration ---
        # Fonts
        self.ui_font_primary = ("Arial", 10)
        self.ui_font_v_grid = ("Arial", 12)
        self.ui_font_q_grid_directional = ("Arial", 9)
        self.ui_font_q_grid_center = ("Arial", 10)

        # Colors
        self.terminal_pos_color = "lightgreen"  # Color for positive reward terminal state
        self.terminal_neg_color = "salmon"      # Color for negative reward terminal state
        self.wall_color = "grey"                # Color for wall cells
        self.default_cell_bg_color = "white"    # Default background for regular grid cells
        self.agent_marker_color = "blue"        # Color for the 'here' agent marker

        # --- Application State ---
        self.current_grid_mode = "v"  # Display mode: "v" for V-values/policy, "q" for Q-values
        self.ql_mode = False # Tracks if a Q-learning based algorithm is active (affects display)
        self.cells = {} # Stores references to grid cell widgets (Labels or Frames)
        self.current_algorithm_instance = None # Holds the currently active RL algorithm instance
        self.interactive_widgets = [] # List of UI widgets to disable/enable during algorithm execution

        # --- Main Frames ---
        # Frame for the grid display
        self.grid_frame = tk.Frame(master)
        self.grid_frame.grid(row=0, column=0, sticky="nsew")

        # Frame for the control panel
        self.control_panel_frame = tk.Frame(master)
        self.control_panel_frame.grid(row=1, column=0, sticky="ew")

        self._setup_control_panel()

        # Configure root window resizing behavior
        master.grid_rowconfigure(0, weight=1) # Grid display area should expand
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=0) # Control panel area has fixed height
        
        self._initialize_v_grid_display() # Initialize with V-grid display

    def _set_controls_enabled(self, enabled_state):
        """
        Enables or disables interactive UI controls stored in self.interactive_widgets.
        Args:
            enabled_state (bool): True to enable controls, False to disable.
        """
        new_state = tk.NORMAL if enabled_state else tk.DISABLED
        for widget in self.interactive_widgets:
            widget.config(state=new_state)

    def _setup_control_panel(self):
        """Sets up the control panel with buttons for algorithms and parameter input fields."""
        
        # Configure column weights for responsiveness in control_panel_frame
        # Control panel has 6 columns (0-5) for layout purposes.
        for i in range(6): 
            self.control_panel_frame.grid_columnconfigure(i, weight=1, minsize=70) 

        # Define padding and font for control panel elements
        pady_buttons = 5; padx_buttons = 5
        pady_params = 3; padx_params = 5
        pady_bottom_row = 5
        ui_font = self.ui_font_primary

        # --- Row 0: Algorithm Execution Buttons ---
        self.value_iteration_button = tk.Button(self.control_panel_frame, text="Run Value Iteration", command=self._run_value_iteration_clicked, font=ui_font)
        self.value_iteration_button.grid(row=0, column=0, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        self.interactive_widgets.append(self.value_iteration_button)

        self.policy_iteration_button = tk.Button(self.control_panel_frame, text="Run Policy Iteration", command=self._run_policy_iteration_clicked, font=ui_font)
        self.policy_iteration_button.grid(row=0, column=1, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        self.interactive_widgets.append(self.policy_iteration_button)

        self.q_learning_button = tk.Button(self.control_panel_frame, text="Run Q-Learning", command=self._run_q_learning_clicked, font=ui_font)
        self.q_learning_button.grid(row=0, column=2, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        self.interactive_widgets.append(self.q_learning_button)

        self.epsilon_greedy_q_button = tk.Button(self.control_panel_frame, text="Run Epsilon Greedy", command=self._run_epsilon_greedy_clicked, font=ui_font)
        self.epsilon_greedy_q_button.grid(row=0, column=3, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        self.interactive_widgets.append(self.epsilon_greedy_q_button)
        
        self.reset_button = tk.Button(self.control_panel_frame, text="Run Decaying E-Greedy", command=lambda: self._run_epsilon_greedy_clicked(decaying=True), font=ui_font)
        self.reset_button.grid(row=0, column=4, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        self.interactive_widgets.append(self.reset_button)

        self.display_button = tk.Button(self.control_panel_frame, text="Cycle Display Mode", command=self._toggle_display_mode_clicked, font=ui_font)
        self.display_button.grid(row=0, column=5, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        # Note: self.display_button is NOT added to interactive_widgets, it remains enabled.
        
        # --- Row 1: MDP Parameters (X Value, R Value, Alpha) ---
        x_value_label = tk.Label(self.control_panel_frame, text="X Value (%):", font=ui_font)
        x_value_label.grid(row=1, column=0, padx=padx_params, pady=pady_params, sticky="e")
        self.x_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.x_value_entry.grid(row=1, column=1, padx=padx_params, pady=pady_params, sticky="w")
        self.x_value_entry.insert(0, "90")
        self.interactive_widgets.append(self.x_value_entry)

        r_value_label = tk.Label(self.control_panel_frame, text="R Value (Reward):", font=ui_font)
        r_value_label.grid(row=1, column=2, padx=padx_params, pady=pady_params, sticky="e")
        self.r_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.r_value_entry.grid(row=1, column=3, padx=padx_params, pady=pady_params, sticky="w")
        self.r_value_entry.insert(0, "-0.04")
        self.interactive_widgets.append(self.r_value_entry)

        a_value_label = tk.Label(self.control_panel_frame, text="Alpha (QL Rate):", font=ui_font)
        a_value_label.grid(row=1, column=4, padx=padx_params, pady=pady_params, sticky="e")
        self.a_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.a_value_entry.grid(row=1, column=5, padx=padx_params, pady=pady_params, sticky="w")
        self.a_value_entry.insert(0, "0.5")
        self.interactive_widgets.append(self.a_value_entry)
        
        # --- Row 2: MDP/Algorithm Parameters (Epsilon, Discount) ---
        epsilon_label = tk.Label(self.control_panel_frame, text="Epsilon (Explore %):", font=ui_font)
        epsilon_label.grid(row=2, column=0, padx=padx_params, pady=pady_params, sticky="e")
        self.epsilon_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.epsilon_entry.grid(row=2, column=1, padx=padx_params, pady=pady_params, sticky="w")
        self.epsilon_entry.insert(0, "0.001")
        self.interactive_widgets.append(self.epsilon_entry)

        discount_label = tk.Label(self.control_panel_frame, text="Discount Factor:")
        discount_label.grid(row=2, column=2, padx=padx_params, pady=pady_params, sticky="e")
        self.discount_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.discount_entry.grid(row=2, column=3, padx=padx_params, pady=pady_params, sticky="w")
        self.discount_entry.insert(0, "0.99")
        self.interactive_widgets.append(self.discount_entry)

        # --- Row 3: Output and Speed Control ---
        speed_slider_label = tk.Label(self.control_panel_frame, text="Speed Multiplier:", font=ui_font)
        speed_slider_label.grid(row=3, column=0, padx=padx_params, pady=pady_bottom_row, sticky="e")
        self.speed_slider = tk.Scale(self.control_panel_frame, from_=.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, font=ui_font) 
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=3, column=1, columnspan=2, padx=padx_params, pady=pady_bottom_row, sticky="ew")
        self.interactive_widgets.append(self.speed_slider)
        
        self.output_label = tk.Label(self.control_panel_frame, text="", width=40, anchor="w", justify=tk.LEFT, font=ui_font) 
        self.output_label.grid(row=3, column=3, columnspan=3, padx=padx_params, pady=pady_bottom_row, sticky="ew")


    def _get_mdp_model_from_ui(self): 
        """
        Retrieves MDP parameters from UI entries, validates them, and creates an MDP object.
        Returns: MDP object if inputs are valid, None otherwise.
        """
        try: r_val = float(self.r_value_entry.get())
        except ValueError: self.set_status_message("Error: R Value (Reward) must be a number."); return None
        try:
            x_val = float(self.x_value_entry.get())
            if not (0 <= x_val <= 100): self.set_status_message("Error: X Value (intended move %) must be between 0 and 100."); return None
        except ValueError: self.set_status_message("Error: X Value must be a number."); return None
        try:
            disc_val = float(self.discount_entry.get())
            if not (0 <= disc_val <= 1): self.set_status_message("Error: Discount Factor must be between 0 and 1."); return None
        except ValueError: self.set_status_message("Error: Discount Factor must be a number."); return None
        try: float(self.a_value_entry.get()) 
        except ValueError: self.set_status_message("Error: Alpha(QL) must be a number."); return None
        try: float(self.epsilon_entry.get()) 
        except ValueError: self.set_status_message("Error: Epsilon (Explore %) must be a number."); return None

        states_list = [(r, c) for r in range(3) for c in range(4)]
        terminal_states_list = [(0, 3), (1, 3)]
        actions_list = ["up", "down", "left", "right"]
        rewards_dict = {}
        for state in states_list:
            if state in terminal_states_list: rewards_dict[state] = {action: 0 for action in actions_list}
            elif state == (1, 1): rewards_dict[state] = {action: 0 for action in actions_list} 
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
                row, col = state; row_diff, col_diff = action_index_difference[action]
                intended_next_state_candidate = (row + row_diff, col + col_diff)
                intended_next_state = state if not (0 <= intended_next_state_candidate[0] < grid_rows and 0 <= intended_next_state_candidate[1] < grid_cols) or intended_next_state_candidate == (1, 1) else intended_next_state_candidate
                transition_dict[state][action][intended_next_state] = transition_dict[state][action].get(intended_next_state, 0.0) + prob_intended
                for side_action in side_actions_map[action]:
                    side_row_diff, side_col_diff = action_index_difference[side_action]
                    side_next_state_candidate = (row + side_row_diff, col + side_col_diff)
                    side_next_state = state if not (0 <= side_next_state_candidate[0] < grid_rows and 0 <= side_next_state_candidate[1] < grid_cols) or side_next_state_candidate == (1, 1) else side_next_state_candidate
                    transition_dict[state][action][side_next_state] = transition_dict[state][action].get(side_next_state, 0.0) + prob_side
        return MDP(states_list, actions_list, rewards_dict, transition_dict, disc_val)

    def _initialize_q_grid_display(self): 
        """Initializes or clears the grid for Q-value display."""
        self.set_status_message("") # Clear any previous status messages
        num_displayable_cells = 9 
        initial_q_quadtuples = [("0.00", "0.00", "0.00", "0.00")] * num_displayable_cells
        self._draw_q_grid_elements(initial_q_quadtuples, q_learn_active=self.ql_mode)

    def _initialize_v_grid_display(self): 
        """Initializes or clears the grid for V-value/Policy display."""
        self.set_status_message("") # Clear any previous status messages
        self.current_grid_mode = "v"
        num_displayable_cells = 9 
        initial_v_tuples = [("0.00", "up")] * num_displayable_cells
        self._draw_v_grid_elements(initial_v_tuples) 

    def _initialize_grid_display(self): 
        """Calls the appropriate grid initialization based on the current display mode."""
        if self.current_grid_mode == "v": self._initialize_v_grid_display()
        else: self._initialize_q_grid_display()

    def refresh_display_from_algorithm_data(self, display_data, iteration_delay_info=None):
        """
        Refreshes the grid display based on data from an algorithm.
        Args:
            display_data (dict): Contains data for display.
            iteration_delay_info (tuple, optional): (should_delay_bool, delay_factor_float).
        """
        v_display_tuples = display_data.get('v_display_tuples')
        q_display_quads = display_data.get('q_display_quads')
        current_agent_state = display_data.get('current_agent_state')
        q_learn_active_flag = display_data.get('q_learn_active', self.ql_mode)

        # Choose drawing method based on current display mode
        if self.current_grid_mode == "v":
            if v_display_tuples is not None:
                 self._draw_v_grid_elements(v_display_tuples, type=display_data.get('display_type'))
            elif q_display_quads is not None: # Fallback for some algorithms that might pass Q-data even in V-mode
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        else: # Q-mode display
            if q_display_quads is not None:
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        
        # Handle iteration delay if specified
        if iteration_delay_info and iteration_delay_info[0]: 
            self.perform_iteration_delay(iteration_delay_info[1])
        else: 
             self.master.update_idletasks() # Ensure UI responsiveness

    def _toggle_display_mode_clicked(self): 
        """Cycles between V-value/Policy display and Q-value display."""
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
            self.set_status_message("Algorithm running. Toggle after completion.") 
            return 

        # Clear existing grid widgets before redrawing
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.cells.clear() 

        # Switch mode and re-initialize the appropriate grid
        if self.current_grid_mode == "v":
            self.current_grid_mode = "q"; self._initialize_q_grid_display()
        else:
            self.current_grid_mode = "v"; self._initialize_v_grid_display()
        
        # If an algorithm instance exists and has data, refresh display in the new mode
        if self.current_algorithm_instance and hasattr(self.current_algorithm_instance, 'get_current_display_data_for_toggle'):
            data = self.current_algorithm_instance.get_current_display_data_for_toggle()
            self.refresh_display_from_algorithm_data(data, iteration_delay_info=(False, 0))
        
        # Restore or clear status message appropriately
        current_status = self.output_label.cget("text")
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
             if not current_status.startswith("Error:"): # Don't overwrite error messages
                self.set_status_message(f"{type(self.current_algorithm_instance).__name__} is running...") 
        elif not (self.current_algorithm_instance and self.current_algorithm_instance.is_running and current_status.startswith("Error:")):
            # Clear status only if no error and no algo running
            if not current_status.startswith("Error:"):
                self.set_status_message("")


    def _draw_v_grid_elements(self, v_policy_tuples_list, type=None): 
        """
        Draws the V-values and policy arrows (or just policy actions) in the grid.
        Args:
            v_policy_tuples_list (list): A list of (v_score_str, direction_str) tuples for the 9 displayable cells.
            type (str, optional): If not None (e.g. "policy_only"), displays only the direction string.
                                  Otherwise, displays formatted V-value and policy.
        """
        if len(v_policy_tuples_list) != 9: return 
        for widget in self.grid_frame.winfo_children(): widget.destroy() # Clear previous grid elements
        self.cells.clear() 
        for i in range(3): self.grid_frame.grid_rowconfigure(i, weight=1, minsize=100)
        for j in range(4): self.grid_frame.grid_columnconfigure(j, weight=1, minsize=100)
        
        tuple_index = 0
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                # Determine cell type for special styling
                is_terminal_pos, is_terminal_neg, is_wall_cell = cell_pos == (0,3), cell_pos == (1,3), cell_pos == (1,1)
                current_text = ""
                bg_color = self.default_cell_bg_color

                if is_terminal_pos: current_text = "1.00"; bg_color = self.terminal_pos_color
                elif is_terminal_neg: current_text = "-1.00"; bg_color = self.terminal_neg_color
                elif is_wall_cell: current_text = ""; bg_color = self.wall_color # Wall cell
                else: # Regular, non-terminal, non-wall cells
                    if tuple_index < len(v_policy_tuples_list):
                        v_score_str, direction_str = v_policy_tuples_list[tuple_index]
                        # Simplified text format for V-mode
                        current_text = f"V: {v_score_str}\nPolicy: {direction_str}" if type is None else direction_str
                        tuple_index += 1
                    else: current_text = "N/A" # Fallback, should not happen

                cell_label = tk.Label(self.grid_frame, text=current_text, relief=tk.SOLID, padx=10, pady=5, width=10, height=5, font=self.ui_font_v_grid, bg=bg_color)
                cell_label.grid(row=row_idx, column=col_idx, sticky="nsew")
                self.cells[cell_pos] = cell_label
                
    def _draw_q_grid_elements(self, quad_tuples_list, q_learn_active=False, current_agent_state=None):
        """
        Draws the Q-values in their respective directional slots within each cell.
        Args:
            quad_tuples_list (list): List of (up_q, right_q, down_q, left_q) string tuples for the 9 displayable cells.
            q_learn_active (bool): True if a Q-learning based algorithm is active (for 'here' marker).
            current_agent_state (tuple, optional): The (row, col) of the agent for 'here' marker.
        """
        if len(quad_tuples_list) != 9: return
        quad_idx = 0
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                is_terminal_pos, is_terminal_neg, is_wall_cell = cell_pos == (0,3), cell_pos == (1,3), cell_pos == (1,1)
                
                # Determine background color based on cell type
                current_bg_color = self.default_cell_bg_color
                if is_terminal_pos: current_bg_color = self.terminal_pos_color
                elif is_terminal_neg: current_bg_color = self.terminal_neg_color
                elif is_wall_cell: current_bg_color = self.wall_color

                # Create cell frame and sub-labels if they don't exist (first draw or after clearing grid)
                if cell_pos not in self.cells or not isinstance(self.cells[cell_pos], dict):
                    cell_frame = tk.Frame(self.grid_frame, relief=tk.SOLID, bd=1, width=100, height=100, bg=current_bg_color)
                    cell_frame.grid(row=row_idx, column=col_idx, sticky="nsew")
                    cell_frame.grid_propagate(False) 
                    for r_s in range(3): cell_frame.grid_rowconfigure(r_s, weight=1) # For centering sub-labels
                    for c_s in range(3): cell_frame.grid_columnconfigure(c_s, weight=1)
                    
                    # Store sub-labels in a dictionary for easy access and updates
                    self.cells[cell_pos] = { 
                        'frame': cell_frame,
                        'center': tk.Label(cell_frame, text="", font=self.ui_font_q_grid_center, bg=current_bg_color),
                        'top': tk.Label(cell_frame, text="0.00", anchor="s", font=self.ui_font_q_grid_directional, bg=current_bg_color), 
                        'right': tk.Label(cell_frame, text="0.00", anchor="w", font=self.ui_font_q_grid_directional, bg=current_bg_color),
                        'bottom': tk.Label(cell_frame, text="0.00", anchor="n", font=self.ui_font_q_grid_directional, bg=current_bg_color), 
                        'left': tk.Label(cell_frame, text="0.00", anchor="e", font=self.ui_font_q_grid_directional, bg=current_bg_color)
                    }
                    
                    # Place sub-labels within the cell_frame
                    self.cells[cell_pos]['center'].grid(row=1, column=1)
                    self.cells[cell_pos]['top'].grid(row=0, column=1, sticky="ew")
                    self.cells[cell_pos]['right'].grid(row=1, column=2, sticky="ns")
                    self.cells[cell_pos]['bottom'].grid(row=2, column=1, sticky="ew")
                    self.cells[cell_pos]['left'].grid(row=1, column=0, sticky="ns")
                else: # If cell frame and labels already exist, just update their background
                    self.cells[cell_pos]['frame'].config(bg=current_bg_color)
                    for lbl_key in ['center','top','right','bottom','left']:
                        self.cells[cell_pos][lbl_key].config(bg=current_bg_color)

                cell_ui = self.cells[cell_pos]
                center_text_val = ""
                center_fg_color = "black" # Default text color for center label

                # Determine text for the center label (agent marker or terminal reward)
                if current_agent_state == cell_pos and self.current_algorithm_instance and \
                   cell_pos not in self.current_algorithm_instance.mdp.terminal_states and not is_wall_cell:
                    center_text_val = "here"
                    center_fg_color = self.agent_marker_color 
                
                # Configure cell display based on its type
                if is_terminal_pos: 
                    cell_ui['center'].config(text="1.00" if not q_learn_active else center_text_val, fg=center_fg_color)
                    for d in ['top','right','bottom','left']: cell_ui[d].config(text="") # Clear Q-values
                elif is_terminal_neg: 
                    cell_ui['center'].config(text="-1.00" if not q_learn_active else center_text_val, fg=center_fg_color)
                    for d in ['top','right','bottom','left']: cell_ui[d].config(text="")
                elif is_wall_cell: 
                    # Background already set to wall_color, clear text from all sub-labels
                    for lbl_key in ['center','top','right','bottom','left']: cell_ui[lbl_key].config(text="")
                else: # Regular, non-terminal, non-wall cells - display Q-values
                    if quad_idx < len(quad_tuples_list):
                        quad_data = quad_tuples_list[quad_idx]
                        cell_ui['top'].config(text=f"{quad_data[0]}")
                        cell_ui['right'].config(text=f"{quad_data[1]}")
                        cell_ui['bottom'].config(text=f"{quad_data[2]}")
                        cell_ui['left'].config(text=f"{quad_data[3]}")
                        cell_ui['center'].config(text=center_text_val, fg=center_fg_color) 
                        quad_idx +=1

    def set_status_message(self, message): 
        """Updates the status message label in the GUI."""
        self.output_label.config(text=message)

    def perform_iteration_delay(self, base_delay=0.2):
        """Handles GUI updates and introduces a delay for visualization."""
        if base_delay and base_delay > 0: self.master.update(); time.sleep(base_delay / float(self.speed_slider.get()))
        else: self.master.update_idletasks() 

    def _run_generic_algorithm(self, algorithm_class, **kwargs):
        """Helper to stop current algorithm and start a new one, managing control states."""
        self._set_controls_enabled(False) 
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
            self.current_algorithm_instance.stop() 
            # Schedule the new algorithm start to allow Tkinter to process the stop event
            self.master.after(50, lambda: self._actually_run_generic_algorithm(algorithm_class, **kwargs))
            return
        self._actually_run_generic_algorithm(algorithm_class, **kwargs)

    def _actually_run_generic_algorithm(self, algorithm_class, **kwargs):
        """Internal helper to run algorithm after ensuring controls are set."""
        self._set_controls_enabled(False) 
        mdp_model = self._get_mdp_model_from_ui()
        if mdp_model is None:  
            self._set_controls_enabled(True) 
            return
        
        self.current_algorithm_instance = algorithm_class(mdp_model, self, **kwargs)
        if hasattr(self.current_algorithm_instance, 'start') and callable(getattr(self.current_algorithm_instance, 'start')):
            self.current_algorithm_instance.start()
        elif hasattr(self.current_algorithm_instance, 'run_algorithm') and callable(getattr(self.current_algorithm_instance, 'run_algorithm')):
            self.current_algorithm_instance.run_algorithm() 

    def _run_value_iteration_clicked(self): self._run_generic_algorithm(ValueIterationAlgorithm)
    def _run_policy_iteration_clicked(self): self._run_generic_algorithm(PolicyIterationAlgorithm)
    def _run_q_learning_clicked(self): self._run_generic_algorithm(QLearningAlgorithm)
    def _run_epsilon_greedy_clicked(self, decaying=False): self._run_generic_algorithm(EpsilonGreedyAlgorithm, decaying_epsilon=decaying)

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

class RLAlgorithm:
    """Base class for Reinforcement Learning algorithms."""
    def __init__(self, mdp_model, app_interface): 
        self.mdp = mdp_model
        self.app = app_interface # Interface to GridworldApp for UI updates
        self.V = {s: 0.0 for s in self.mdp.algo_states} 
        self.Q = {s: {a: 0.0 for a in self.mdp.actions} for s in self.mdp.algo_states} 
        self.policy = {s: None for s in self.mdp.algo_states} 
        self.is_running = False 
        self.iteration = 0      
        self.max_iterations = 2000 

    def start(self): 
        """Starts the algorithm. Should be overridden by subclasses for non-blocking execution."""
        self.is_running = True; self.iteration = 0
        # self.app._set_controls_enabled(False) # Now handled by _run_generic_algorithm
        raise NotImplementedError("Subclasses must implement the start method.")
        
    def stop(self):
        """Stops the current algorithm execution and re-enables UI controls."""
        self.is_running = False
        self.app._set_controls_enabled(True)

    def _prepare_v_display_tuples(self):
        """Prepares V-values and policy for display in V-mode."""
        tuples = []
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        for state_coord in ordered_display_cells:
            tuples.append((f"{self.V.get(state_coord, 0.0):.2f}", self.policy.get(state_coord, "")))
        return tuples
    
    def _prepare_q_display_quads(self, q_data_source=None):
        """Prepares Q-values for display in Q-mode."""
        quads = []
        source = q_data_source if q_data_source is not None else self.Q
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        for state_coord in ordered_display_cells:
            q_s_vals = source.get(state_coord, {}) 
            quads.append((f"{q_s_vals.get('up', 0.0):.2f}", f"{q_s_vals.get('right', 0.0):.2f}",
                          f"{q_s_vals.get('down', 0.0):.2f}", f"{q_s_vals.get('left', 0.0):.2f}"))
        return quads

    def get_current_display_data_for_toggle(self):
        """
        Prepares data for display when toggling modes.
        For VI/PI, derives Q-values from V. For QL/EG, derives policy from Q.
        """
        v_tuples = self._prepare_v_display_tuples()
        q_for_display_dict = {} 
        if isinstance(self, (ValueIterationAlgorithm, PolicyIterationAlgorithm)):
            for s in self.mdp.algo_states: 
                if s in self.mdp.terminal_states: 
                    q_for_display_dict[s] = {a: self.V.get(s,0.0) for a in self.mdp.actions} 
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
        else: 
            q_display_quads = self._prepare_q_display_quads()
            for s_pol in self.mdp.algo_states: 
                if s_pol not in self.mdp.terminal_states and self.Q.get(s_pol) and self.Q[s_pol]:
                    best_action = max(self.Q[s_pol], key=self.Q[s_pol].get, default=None)
                    if best_action: self.policy[s_pol] = best_action
        return {'v_display_tuples': v_tuples, 'q_display_quads': q_display_quads, 
                'current_agent_state': getattr(self, 'current_state', None), 
                'q_learn_active': isinstance(self, (QLearningAlgorithm, EpsilonGreedyAlgorithm))}

class ValueIterationAlgorithm(RLAlgorithm):
    """Implements the Value Iteration algorithm."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        if (0,3) in self.V: self.V[(0,3)] = self.mdp.rewards.get((0,3),{}).get("up", 1.0) 
        if (1,3) in self.V: self.V[(1,3)] = self.mdp.rewards.get((1,3),{}).get("up", -1.0)

    def start(self):
        """Starts the Value Iteration process, run asynchronously with Tkinter."""
        self.is_running = True; self.iteration = 0
        self.app._initialize_grid_display(); self.app.ql_mode = False 
        self._perform_one_iteration()

    def _perform_one_iteration(self):
        """Performs a single iteration of Value Iteration and schedules the next if needed."""
        if not self.is_running: self.app._set_controls_enabled(True); return 
        
        try: epsilon = float(self.app.epsilon_entry.get())
        except ValueError: self.app.set_status_message("Error: Epsilon must be a number."); self.stop(); return
        
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
            if s_policy not in self.mdp.terminal_states and q_values_for_v_iteration.get(s_policy) and q_values_for_v_iteration[s_policy]:
                best_action = max(q_values_for_v_iteration[s_policy], key=q_values_for_v_iteration[s_policy].get, default=None)
                if best_action: self.policy[s_policy] = best_action
        v_tuples, q_quads = self._prepare_flat_display_data_from_v_and_q_dict(q_values_for_v_iteration)
        self.app.refresh_display_from_algorithm_data({'v_display_tuples': v_tuples, 'q_display_quads': q_quads, 'q_learn_active': False}, iteration_delay_info=(False, 0)) 
        self.iteration +=1
        if delta > threshold and self.iteration < self.max_iterations:
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000) 
            self.app.master.after(delay_ms, self._perform_one_iteration)
        else:
            status = 'converged' if delta <= threshold else 'stopped (max iterations)'
            self.app.set_status_message(f"Value iteration {status} after {self.iteration} iterations.")
            self.stop() 

class PolicyIterationAlgorithm(RLAlgorithm):
    """Implements the Policy Iteration algorithm."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        if (0,3) in self.V: self.V[(0,3)] = self.mdp.rewards.get((0,3),{}).get("up",1.0)
        if (1,3) in self.V: self.V[(1,3)] = self.mdp.rewards.get((1,3),{}).get("up",-1.0)
        for s_init in self.mdp.algo_states:
            if s_init not in self.mdp.terminal_states: self.policy[s_init] = random.choice(self.mdp.actions) 

    def start(self):
        """Starts the Policy Iteration process, run asynchronously with Tkinter."""
        self.is_running = True; self.iteration = 0
        self.app._initialize_grid_display(); self.app.ql_mode = False
        self._perform_one_policy_iteration_step()

    def _policy_evaluation(self, epsilon_eval): 
        """Performs the policy evaluation step (iterative)."""
        threshold_eval = epsilon_eval * (1 - self.mdp.discount_factor) / self.mdp.discount_factor if self.mdp.discount_factor != 1.0 else epsilon_eval
        eval_V = self.V.copy() 
        while True: 
            delta_eval = 0; new_eval_V = eval_V.copy()
            for s_eval in self.mdp.algo_states:
                if s_eval in self.mdp.terminal_states: continue
                old_v_s_eval = eval_V[s_eval]; action_to_eval = self.policy[s_eval]
                if action_to_eval is None: new_eval_V[s_eval] = 0; continue 
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
        """Performs one step of policy iteration (evaluation + improvement) and schedules the next."""
        if not self.is_running: self.app._set_controls_enabled(True); return
        try: epsilon = float(self.app.epsilon_entry.get())
        except ValueError: self.app.set_status_message("Error: Epsilon must be a number."); self.stop(); return

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
            if action_q_s_improve: # Ensure dict is not empty
                best_action_improve = max(action_q_s_improve, key=action_q_s_improve.get, default=None)
                if best_action_improve: self.policy[s_improve] = best_action_improve
                if best_action_improve != old_action_improve: policy_stable = False
        
        v_tuples_pi, q_quads_pi = self._prepare_flat_display_data_from_v_and_q_dict(q_for_improvement)
        self.app.refresh_display_from_algorithm_data({'v_display_tuples': v_tuples_pi, 'q_display_quads': q_quads_pi, 'q_learn_active': False}, iteration_delay_info=(False, 0))
        self.iteration +=1
        if not policy_stable and self.iteration < self.max_iterations :
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000)
            self.app.master.after(delay_ms, self._perform_one_policy_iteration_step)
        else:
            status = 'converged' if policy_stable else 'stopped (max iterations)'
            self.app.set_status_message(f"Policy iteration {status} after {self.iteration} iterations.")
            self.stop()
    
    def _prepare_flat_display_data_from_v_and_q_dict(self, q_values_dict):
        """Prepares V and Q display data from a Q-values dictionary (used in VI/PI)."""
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
    """Implements user-interactive Q-Learning."""
    def __init__(self, mdp_model, app_interface):
        super().__init__(mdp_model, app_interface)
        self.N_sa = {s: {a: 0 for a in self.mdp.actions} for s in self.mdp.algo_states} # State-action visit counts
        self.current_state = (2,0) # Agent's starting position
        self.move_var = tk.StringVar() # For capturing user key presses
        for terminal_s in self.mdp.terminal_states: 
            if terminal_s in self.Q:
                for a_term in self.mdp.actions: self.Q[terminal_s][a_term] = 0.0 

    def _on_key(self, event):
        """Handles key press events for agent movement."""
        if event.keysym in ['Up', 'Down', 'Left', 'Right']: self.move_var.set(event.keysym.lower())

    def run_algorithm(self): 
        """Runs the Q-learning algorithm, driven by user key presses. This is a blocking loop."""
        self.is_running = True 
        self.app.ql_mode = True; self.app.current_grid_mode = 'q'; self.app._initialize_q_grid_display()
        try: initial_alpha = float(self.app.a_value_entry.get())
        except ValueError: self.app.set_status_message("Error: Alpha(QL) must be a number."); self.stop(); return
        
        move_count = 0; max_moves = 40 
        self.app.master.bind('<Key>', self._on_key)
        self.app.set_status_message("Use the arrow keys to move the marker.")
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))

        while move_count < max_moves:
            self.app.master.wait_variable(self.move_var); action = self.move_var.get(); self.move_var.set("")
            if not self.is_running: break 
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
        self.stop() 

class EpsilonGreedyAlgorithm(QLearningAlgorithm): 
    """Implements Epsilon-Greedy Q-Learning (automated steps), run asynchronously."""
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
        """Performs a single step of Epsilon-Greedy Q-Learning and schedules the next."""
        if not self.is_running or self.move_count >= self.max_moves:
            if self.is_running : self.app.set_status_message(f"Epsilon-Greedy finished after {self.move_count} moves.")
            self.stop(); return

        try: 
            initial_alpha = float(self.app.a_value_entry.get())
            epsilon_val = float(self.app.epsilon_entry.get())
        except ValueError:
            self.app.set_status_message("Error: Alpha or Epsilon is not a valid number.")
            self.stop(); return

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
            next_step_delay_after = 500 
        self.app.master.after(next_step_delay_after, self._perform_one_eg_step)
    
def main():
    """Main function to create and run the Tkinter application."""
    root = tk.Tk()
    app = GridworldApp(root) 
    root.mainloop() 

if __name__ == "__main__":
    main()

[end of src/mdp_rl_gridworld.py]
