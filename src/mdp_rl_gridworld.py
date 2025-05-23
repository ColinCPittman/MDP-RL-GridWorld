#!/usr/bin/env python
# coding: utf-8

# # Imports and Gloabls

# In[34]:


import tkinter as tk
import time
import random
from mdp.model import MDP

# # Class Definitions

class GridworldApp:
    """
    Manages the main Tkinter application window, GUI components, user interactions,
    and the overall orchestration of the Gridworld Reinforcement Learning environment.

    This class is responsible for:
    - Setting up the main UI layout including the grid display and control panel.
    - Handling user inputs for MDP parameters and algorithm selection.
    - Managing the execution and display of different RL algorithms.
    - Updating the GUI to reflect algorithm progress and results.
    """
    def __init__(self, master):
        """
        Initializes the GridworldApp application.

        Args:
            master (tk.Tk): The root Tkinter window for the application.
        """
        self.master = master  # The main Tkinter window
        master.title("Gridworld Display")

        # --- UI Configuration ---
        # Define fonts for various UI elements
        self.ui_font_primary = ("Arial", 10)  # General UI font
        self.ui_font_v_grid = ("Arial", 12)   # Font for V-value and policy display in grid cells
        self.ui_font_q_grid_directional = ("Arial", 9) # Font for Q-value directional text
        self.ui_font_q_grid_center = ("Arial", 10)    # Font for Q-value center text (e.g., 'here' marker)

        # Define colors for different grid cell types and UI elements
        self.terminal_pos_color = "lightgreen"  # Background for positive reward terminal state cells
        self.terminal_neg_color = "salmon"      # Background for negative reward terminal state cells
        self.wall_color = "grey"                # Background for wall cells
        self.default_cell_bg_color = "white"    # Default background for regular (non-terminal, non-wall) grid cells
        self.agent_marker_color = "blue"        # Text color for the 'here' agent marker in Q-learning display

        # --- Application State Variables ---
        self.current_grid_mode = "v"  # Current display mode: "v" for V-values/policy, "q" for Q-values
        self.ql_mode = False          # Boolean flag indicating if a Q-learning based algorithm is currently active.
                                      # This affects display elements like the 'here' marker.
        self.cells = {}               # Dictionary to store references to Tkinter grid cell widgets (Labels or Frames).
                                      # Keys are (row, col) tuples.
        self.current_algorithm_instance = None # Holds the instance of the currently executing RL algorithm.
                                               # This allows for stopping/interacting with the active algorithm.
        self.interactive_widgets = [] # List to store references to UI widgets (buttons, entries, sliders)
                                      # that should be disabled during algorithm execution to prevent conflicts.

        # --- Main UI Frames ---
        # Frame for the grid display (where V-values or Q-values are shown)
        self.grid_frame = tk.Frame(master)
        self.grid_frame.grid(row=0, column=0, sticky="nsew") # Sticky makes it expand with window resize

        # Frame for the control panel (buttons, parameter inputs, status messages)
        self.control_panel_frame = tk.Frame(master)
        self.control_panel_frame.grid(row=1, column=0, sticky="ew") # Sticky to expand horizontally

        self._setup_control_panel()  # Initialize the widgets within the control panel

        # Configure resizing behavior for the root window's grid layout
        master.grid_rowconfigure(0, weight=1)    # Grid display area (row 0) should expand vertically
        master.grid_columnconfigure(0, weight=1) # Grid display area (column 0) should expand horizontally
        master.grid_rowconfigure(1, weight=0)    # Control panel area (row 1) has a fixed height

        self._initialize_v_grid_display() # Initialize the grid display with V-values/Policy by default

    def _set_controls_enabled(self, enabled_state):
        """
        Enables or disables interactive UI controls.

        This method iterates through `self.interactive_widgets` (which includes
        algorithm buttons, parameter entries, and the speed slider) and sets their
        state to normal (enabled) or disabled.

        Args:
            enabled_state (bool): True to enable controls, False to disable.
        """
        new_state = tk.NORMAL if enabled_state else tk.DISABLED
        for widget in self.interactive_widgets:
            widget.config(state=new_state)

    def _setup_control_panel(self):
        """
        Sets up the control panel with algorithm execution buttons,
        MDP parameter input fields, and other controls.
        """
        
        # Configure column weights for responsive layout within the control_panel_frame.
        # The control panel is conceptually divided into 6 columns for arranging widgets.
        # Giving them weight ensures they resize proportionally if the window width changes.
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
        self.interactive_widgets.append(self.reset_button) # Decaying E-Greedy button

        # Button to cycle between V-value/Policy display and Q-value display
        self.display_button = tk.Button(self.control_panel_frame, text="Cycle Display Mode", command=self._toggle_display_mode_clicked, font=ui_font)
        self.display_button.grid(row=0, column=5, padx=padx_buttons, pady=pady_buttons, sticky="ew")
        # Note: self.display_button is intentionally NOT added to self.interactive_widgets,
        # as it should remain enabled even when an algorithm is running to allow display mode changes.
        
        # --- Row 1: MDP Parameters (X Value for transition probability, R Value for step reward, Alpha for Q-learning rate) ---
        # X Value (Probability of intended move)
        x_value_label = tk.Label(self.control_panel_frame, text="X Value (%):", font=ui_font) # Label for X Value
        x_value_label.grid(row=1, column=0, padx=padx_params, pady=pady_params, sticky="e")
        self.x_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font)
        self.x_value_entry.grid(row=1, column=1, padx=padx_params, pady=pady_params, sticky="w")
        self.x_value_entry.insert(0, "90") # Default X Value
        self.interactive_widgets.append(self.x_value_entry)

        # R Value (Reward for non-terminal steps)
        r_value_label = tk.Label(self.control_panel_frame, text="R Value (Reward):", font=ui_font) # Label for R Value
        r_value_label.grid(row=1, column=2, padx=padx_params, pady=pady_params, sticky="e")
        self.r_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font) # Entry field for R Value
        self.r_value_entry.grid(row=1, column=3, padx=padx_params, pady=pady_params, sticky="w")
        self.r_value_entry.insert(0, "-0.04") # Default R Value
        self.interactive_widgets.append(self.r_value_entry)

        # Alpha (Learning Rate for Q-Learning)
        a_value_label = tk.Label(self.control_panel_frame, text="Alpha (QL Rate):", font=ui_font) # Label for Alpha
        a_value_label.grid(row=1, column=4, padx=padx_params, pady=pady_params, sticky="e")
        self.a_value_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font) # Entry field for Alpha
        self.a_value_entry.grid(row=1, column=5, padx=padx_params, pady=pady_params, sticky="w")
        self.a_value_entry.insert(0, "0.5") # Default Alpha value
        self.interactive_widgets.append(self.a_value_entry)
        
        # --- Row 2: MDP/Algorithm Parameters (Epsilon for exploration, Discount factor gamma) ---
        # Epsilon (Exploration probability for Epsilon-Greedy Q-Learning)
        epsilon_label = tk.Label(self.control_panel_frame, text="Epsilon (Explore %):", font=ui_font) # Label for Epsilon
        epsilon_label.grid(row=2, column=0, padx=padx_params, pady=pady_params, sticky="e")
        self.epsilon_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font) # Entry field for Epsilon
        self.epsilon_entry.grid(row=2, column=1, padx=padx_params, pady=pady_params, sticky="w")
        self.epsilon_entry.insert(0, "0.001") # Default Epsilon value (used for VI/PI convergence threshold as well)
        self.interactive_widgets.append(self.epsilon_entry)

        # Discount Factor (Gamma)
        discount_label = tk.Label(self.control_panel_frame, text="Discount Factor:") # Label for Discount Factor
        discount_label.grid(row=2, column=2, padx=padx_params, pady=pady_params, sticky="e")
        self.discount_entry = tk.Entry(self.control_panel_frame, width=5, font=ui_font) # Entry field for Discount Factor
        self.discount_entry.grid(row=2, column=3, padx=padx_params, pady=pady_params, sticky="w")
        self.discount_entry.insert(0, "0.99") # Default Discount Factor
        self.interactive_widgets.append(self.discount_entry)

        # --- Row 3: Output Label and Speed Control Slider ---
        # Speed Multiplier Slider (controls delay in algorithm visualization)
        speed_slider_label = tk.Label(self.control_panel_frame, text="Speed Multiplier:", font=ui_font) # Label for Speed Slider
        speed_slider_label.grid(row=3, column=0, padx=padx_params, pady=pady_bottom_row, sticky="e")
        self.speed_slider = tk.Scale(self.control_panel_frame, from_=.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, font=ui_font) 
        self.speed_slider.set(1.0) # Default speed multiplier
        self.speed_slider.grid(row=3, column=1, columnspan=2, padx=padx_params, pady=pady_bottom_row, sticky="ew")
        self.interactive_widgets.append(self.speed_slider)
        
        # Output Label (for status messages, errors, and algorithm completion info)
        self.output_label = tk.Label(self.control_panel_frame, text="", width=40, anchor="w", justify=tk.LEFT, font=ui_font) 
        self.output_label.grid(row=3, column=3, columnspan=3, padx=padx_params, pady=pady_bottom_row, sticky="ew")


    def _get_mdp_model_from_ui(self): 
        """
        Retrieves MDP parameters from the UI input fields, validates them,
        and creates an initialized MDP object.

        Validates R Value, X Value (transition probability), Discount Factor for numeric types
        and specific ranges. Alpha and Epsilon are validated for numeric type only here;
        their specific validation/usage is handled by the algorithms.

        Returns:
            mdp.model.MDP: An initialized MDP object if all inputs are valid.
            None: If any input is invalid, sets an error message on `self.output_label` and returns None.
        """
        # Validate R Value (Reward for non-terminal step)
        try: r_val = float(self.r_value_entry.get())
        except ValueError: self.set_status_message("Error: R Value (Reward) must be a number."); return None
        
        # Validate X Value (Probability of intended move)
        try:
            x_val = float(self.x_value_entry.get())
            if not (0 <= x_val <= 100): # X Value must be a percentage
                self.set_status_message("Error: X Value (intended move %) must be between 0 and 100."); return None
        except ValueError: self.set_status_message("Error: X Value must be a number."); return None
        
        # Validate Discount Factor (gamma)
        try:
            disc_val = float(self.discount_entry.get())
            if not (0 <= disc_val <= 1): # Discount factor must be between 0 and 1
                self.set_status_message("Error: Discount Factor must be between 0 and 1."); return None
        except ValueError: self.set_status_message("Error: Discount Factor must be a number."); return None
        
        # Validate Alpha (Q-Learning rate) - just for numeric type here
        try: float(self.a_value_entry.get()) 
        except ValueError: self.set_status_message("Error: Alpha(QL) must be a number."); return None
        
        # Validate Epsilon (Exploration rate / VI & PI convergence threshold) - just for numeric type here
        try: float(self.epsilon_entry.get()) 
        except ValueError: self.set_status_message("Error: Epsilon (Explore %) must be a number."); return None

        # Create an MDP instance using the default constructor
        mdp_model = MDP()
        # Initialize the MDP with Pittman Gridworld specific parameters from UI
        # This uses the centralized initialization logic in the MDP class.
        mdp_model.initialize_pittman_gridworld_model(
            x_prob_value=x_val, 
            reward_step_cost=r_val, 
            discount_factor_from_gui=disc_val
        )
        return mdp_model

    def _initialize_q_grid_display(self): 
        """
        Initializes or clears the grid display for Q-values.
        It sets up a blank Q-value grid, typically called when switching to Q-mode
        or at the beginning of a Q-learning algorithm.
        """
        self.set_status_message("") # Clear any previous status messages from the output label
        num_displayable_cells = 9 # There are 9 non-wall cells in the 3x4 grid
        # Prepare a list of default Q-value tuples (all "0.00") for initial display
        initial_q_quadtuples = [("0.00", "0.00", "0.00", "0.00")] * num_displayable_cells
        # Draw the Q-grid elements with these initial values
        self._draw_q_grid_elements(initial_q_quadtuples, q_learn_active=self.ql_mode)

    def _initialize_v_grid_display(self): 
        """
        Initializes or clears the grid display for V-values and Policies.
        It sets up a blank V-value/policy grid, typically called when switching to V-mode
        or at the beginning of Value Iteration or Policy Iteration.
        """
        self.set_status_message("") # Clear any previous status messages
        self.current_grid_mode = "v" # Ensure the application mode is set to V-mode
        num_displayable_cells = 9 # 9 non-wall cells
        # Prepare a list of default V-value and policy tuples for initial display
        initial_v_tuples = [("0.00", "up")] * num_displayable_cells # Default V-value "0.00", default policy "up"
        # Draw the V-grid elements with these initial values
        self._draw_v_grid_elements(initial_v_tuples) 

    def _initialize_grid_display(self): 
        """
        Calls the appropriate grid initialization method based on the current display mode (`self.current_grid_mode`).
        """
        if self.current_grid_mode == "v":
            self._initialize_v_grid_display()
        else: # current_grid_mode == "q"
            self._initialize_q_grid_display()

    def refresh_display_from_algorithm_data(self, display_data, iteration_delay_info=None):
        """
        Refreshes the grid display based on data received from an active RL algorithm.

        This method is called by algorithms during their execution to update the UI
        with new V-values, Q-values, policies, or agent positions.

        Args:
            display_data (dict): A dictionary containing the data to be displayed. Expected keys might include:
                                 'v_display_tuples' (for V-mode),
                                 'q_display_quads' (for Q-mode),
                                 'current_agent_state' (for Q-learning agent marker),
                                 'q_learn_active' (bool, influences Q-mode display for terminals),
                                 'display_type' (optional, for V-mode, e.g., "policy_only").
            iteration_delay_info (tuple, optional): A tuple `(should_delay, delay_duration_factor)`
                                                   to control visualization speed. If `should_delay` is True,
                                                   a delay is introduced. `delay_duration_factor` can be
                                                   used by `perform_iteration_delay`.
        """
        # Extract data from the display_data dictionary
        v_display_tuples = display_data.get('v_display_tuples')
        q_display_quads = display_data.get('q_display_quads')
        current_agent_state = display_data.get('current_agent_state')
        q_learn_active_flag = display_data.get('q_learn_active', self.ql_mode) # Default to current app ql_mode

        # Choose the appropriate drawing method based on the current grid display mode
        if self.current_grid_mode == "v": # If in V-value/Policy display mode
            if v_display_tuples is not None:
                 self._draw_v_grid_elements(v_display_tuples, type=display_data.get('display_type'))
            # Fallback: if V-tuples are not available but Q-quads are, draw Q-grid (might indicate mixed data)
            elif q_display_quads is not None: 
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        else: # If in Q-value display mode
            if q_display_quads is not None:
                self._draw_q_grid_elements(q_display_quads, q_learn_active=q_learn_active_flag, current_agent_state=current_agent_state)
        
        # Handle iteration delay for visualization speed control
        if iteration_delay_info and iteration_delay_info[0]: # If should_delay is True
            self.perform_iteration_delay(iteration_delay_info[1]) # Use the provided delay factor
        else: 
             self.master.update_idletasks() # Ensure UI responsiveness if no delay is applied

    def _toggle_display_mode_clicked(self): 
        """
        Handles the click event for the "Cycle Display Mode" button.
        Switches the grid display between V-value/Policy mode and Q-value mode.
        If an algorithm has been run and data exists, it attempts to refresh the
        display in the new mode using that data.
        """
        # Prevent toggling if an algorithm is currently executing its iterative process
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
            self.set_status_message("Algorithm running. Toggle after completion.") 
            return 

        # Clear existing grid widgets from the grid_frame before redrawing in the new mode
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.cells.clear() # Clear the stored references to cell widgets

        # Switch the display mode and re-initialize the grid for the new mode
        if self.current_grid_mode == "v":
            self.current_grid_mode = "q"
            self._initialize_q_grid_display() # Initialize with empty Q-values
        else: # current_grid_mode == "q"
            self.current_grid_mode = "v"
            self._initialize_v_grid_display() # Initialize with empty V-values/policies
        
        # If an algorithm instance exists (meaning an algorithm was run or is paused)
        # and it has a method to provide its current data, refresh the display.
        if self.current_algorithm_instance and hasattr(self.current_algorithm_instance, 'get_current_display_data_for_toggle'):
            data = self.current_algorithm_instance.get_current_display_data_for_toggle()
            self.refresh_display_from_algorithm_data(data, iteration_delay_info=(False, 0)) # No delay for toggle
        
        # Restore or clear the status message in the output label appropriately
        current_status = self.output_label.cget("text")
        # If an algorithm is (conceptually) running (e.g. QL paused for input)
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
             if not current_status.startswith("Error:"): # Don't overwrite existing error messages
                self.set_status_message(f"{type(self.current_algorithm_instance).__name__} is running...") 
        # If no algorithm is running and there's no persistent error message
        elif not (self.current_algorithm_instance and self.current_algorithm_instance.is_running and current_status.startswith("Error:")):
            if not current_status.startswith("Error:"): # Don't clear error messages
                self.set_status_message("") # Clear status message


    def _draw_v_grid_elements(self, v_policy_tuples_list, type=None): 
        """
        Draws the V-values and corresponding policy directions in the grid cells.

        This method clears the existing grid and redraws all cells. It's used when
        displaying results from Value Iteration or Policy Iteration, or when
        converting Q-values to a policy display.

        Args:
            v_policy_tuples_list (list): A list of (v_score_str, direction_str) tuples.
                                         The list should contain 9 tuples, corresponding to the
                                         9 displayable (non-wall) cells, ordered row by row.
            type (str, optional): If "policy_only" (or any non-None value), displays only
                                  the direction string. Otherwise, displays formatted V-value and policy.
        """
        if len(v_policy_tuples_list) != 9: return # Expect 9 tuples for the 9 displayable cells
        
        # Clear previous grid elements to prevent overlap or stale data
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.cells.clear() # Clear stored cell widget references

        # Configure grid rows and columns to have equal weight and a minimum size for consistent appearance
        for i in range(3): self.grid_frame.grid_rowconfigure(i, weight=1, minsize=100) # 3 rows
        for j in range(4): self.grid_frame.grid_columnconfigure(j, weight=1, minsize=100) # 4 columns
        
        tuple_index = 0 # Index for accessing v_policy_tuples_list
        # Iterate through the 3x4 grid layout
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                
                # Determine cell type for special styling and content
                is_terminal_pos = (cell_pos == (0,3)) # Positive reward terminal state
                is_terminal_neg = (cell_pos == (1,3)) # Negative reward terminal state
                is_wall_cell    = (cell_pos == (1,1)) # Wall state
                
                current_text = "" # Text to be displayed in the cell
                bg_color = self.default_cell_bg_color # Default background color

                if is_terminal_pos:
                    current_text = "1.00" # Fixed value for positive terminal state
                    bg_color = self.terminal_pos_color
                elif is_terminal_neg:
                    current_text = "-1.00" # Fixed value for negative terminal state
                    bg_color = self.terminal_neg_color
                elif is_wall_cell:
                    current_text = "" # No text for wall cells
                    bg_color = self.wall_color
                else: # Regular, non-terminal, non-wall cells
                    if tuple_index < len(v_policy_tuples_list):
                        v_score_str, direction_str = v_policy_tuples_list[tuple_index]
                        # Format text based on 'type' argument
                        if type is None: # Standard display: V-value and policy direction
                            current_text = f"V: {v_score_str}\nPolicy: {direction_str}"
                        else: # Policy_only display: just the direction
                            current_text = direction_str
                        tuple_index += 1
                    else:
                        current_text = "N/A" # Fallback if tuple_list is too short (should not happen)

                # Create and place the label widget for the cell
                cell_label = tk.Label(self.grid_frame, text=current_text, relief=tk.SOLID, 
                                      padx=10, pady=5, width=10, height=5, 
                                      font=self.ui_font_v_grid, bg=bg_color)
                cell_label.grid(row=row_idx, column=col_idx, sticky="nsew") # sticky="nsew" makes label fill cell
                self.cells[cell_pos] = cell_label # Store reference to the label
                
    def _draw_q_grid_elements(self, quad_tuples_list, q_learn_active=False, current_agent_state=None):
        """
        Draws the Q-values in their respective directional slots within each grid cell.
        Each cell (except walls/terminals) is divided to show four Q-values (up, right, down, left)
        and a central area for agent marker or terminal value.

        Args:
            quad_tuples_list (list): A list of (q_up, q_right, q_down, q_left) string tuples.
                                     Should contain 9 tuples for the 9 displayable cells.
            q_learn_active (bool): True if a Q-learning based algorithm is active. This affects
                                   how terminal states are displayed (value vs. potential 'here' marker).
            current_agent_state (tuple, optional): The (row, col) of the agent, used to display
                                                   a 'here' marker in the corresponding cell.
        """
        if len(quad_tuples_list) != 9: return # Expect 9 quad_tuples for the 9 displayable cells
        
        quad_idx = 0 # Index for accessing quad_tuples_list
        # Iterate through the 3x4 grid layout
        for row_idx in range(3):
            for col_idx in range(4):
                cell_pos = (row_idx, col_idx)
                
                # Determine cell type
                is_terminal_pos = (cell_pos == (0,3))
                is_terminal_neg = (cell_pos == (1,3))
                is_wall_cell    = (cell_pos == (1,1))
                
                # Determine background color based on cell type
                current_bg_color = self.default_cell_bg_color
                if is_terminal_pos: current_bg_color = self.terminal_pos_color
                elif is_terminal_neg: current_bg_color = self.terminal_neg_color
                elif is_wall_cell: current_bg_color = self.wall_color

                # Create cell frame and sub-labels (for Q-values) if they don't exist.
                # This happens on the first draw or if the grid was cleared (e.g., by _toggle_display_mode_clicked).
                if cell_pos not in self.cells or not isinstance(self.cells[cell_pos], dict):
                    # Main frame for the cell, helps in structuring sub-labels
                    cell_frame = tk.Frame(self.grid_frame, relief=tk.SOLID, bd=1, width=100, height=100, bg=current_bg_color)
                    cell_frame.grid(row=row_idx, column=col_idx, sticky="nsew")
                    cell_frame.grid_propagate(False) # Prevent frame from shrinking to fit content
                    
                    # Configure internal grid of the cell_frame (3x3 for Q-values and center)
                    for r_s in range(3): cell_frame.grid_rowconfigure(r_s, weight=1) # Rows for top, center, bottom Qs
                    for c_s in range(3): cell_frame.grid_columnconfigure(c_s, weight=1) # Cols for left, center, right Qs
                    
                    # Create and store labels for Q-values (top, right, bottom, left) and center text
                    self.cells[cell_pos] = { 
                        'frame': cell_frame,
                        'center': tk.Label(cell_frame, text="", font=self.ui_font_q_grid_center, bg=current_bg_color),
                        'top': tk.Label(cell_frame, text="0.00", anchor="s", font=self.ui_font_q_grid_directional, bg=current_bg_color), 
                        'right': tk.Label(cell_frame, text="0.00", anchor="w", font=self.ui_font_q_grid_directional, bg=current_bg_color),
                        'bottom': tk.Label(cell_frame, text="0.00", anchor="n", font=self.ui_font_q_grid_directional, bg=current_bg_color), 
                        'left': tk.Label(cell_frame, text="0.00", anchor="e", font=self.ui_font_q_grid_directional, bg=current_bg_color)
                    }
                    
                    # Place sub-labels within the cell_frame's internal grid
                    self.cells[cell_pos]['center'].grid(row=1, column=1)
                    self.cells[cell_pos]['top'].grid(row=0, column=1, sticky="ew")    # Q-up
                    self.cells[cell_pos]['right'].grid(row=1, column=2, sticky="ns")  # Q-right
                    self.cells[cell_pos]['bottom'].grid(row=2, column=1, sticky="ew") # Q-down
                    self.cells[cell_pos]['left'].grid(row=1, column=0, sticky="ns")   # Q-left
                else: 
                    # If cell widgets already exist, just update their background color (e.g., if toggling modes rapidly)
                    self.cells[cell_pos]['frame'].config(bg=current_bg_color)
                    for lbl_key in ['center','top','right','bottom','left']:
                        self.cells[cell_pos][lbl_key].config(bg=current_bg_color)

                # Get the dictionary of UI elements for the current cell
                cell_ui = self.cells[cell_pos]
                center_text_val = "" # Default text for the center label
                center_fg_color = "black" 

                # Determine text for the center label (agent 'here' marker or terminal state value)
                # Show 'here' if Q-learning is active, agent is in this cell, and it's not a terminal/wall state
                if current_agent_state == cell_pos and \
                   self.current_algorithm_instance and \
                   cell_pos not in self.current_algorithm_instance.mdp.terminal_states and \
                   not is_wall_cell:
                    center_text_val = "here"
                    center_fg_color = self.agent_marker_color 
                
                # Configure cell display based on its type (terminal, wall, or regular)
                if is_terminal_pos: 
                    # Positive terminal state: show "1.00" unless Q-learning is active and 'here' marker needs to be shown
                    cell_ui['center'].config(text="1.00" if not q_learn_active else center_text_val, fg=center_fg_color)
                    for d_label in ['top','right','bottom','left']: cell_ui[d_label].config(text="") # Clear Q-value texts
                elif is_terminal_neg: 
                    # Negative terminal state: show "-1.00" similarly
                    cell_ui['center'].config(text="-1.00" if not q_learn_active else center_text_val, fg=center_fg_color)
                    for d_label in ['top','right','bottom','left']: cell_ui[d_label].config(text="")
                elif is_wall_cell: 
                    # Wall cell: clear text from all sub-labels
                    for lbl_key in ['center','top','right','bottom','left']: cell_ui[lbl_key].config(text="")
                else: # Regular, non-terminal, non-wall cells: display Q-values
                    if quad_idx < len(quad_tuples_list):
                        quad_data = quad_tuples_list[quad_idx] # (q_up, q_right, q_down, q_left)
                        cell_ui['top'].config(text=f"{quad_data[0]}")
                        cell_ui['right'].config(text=f"{quad_data[1]}")
                        cell_ui['bottom'].config(text=f"{quad_data[2]}")
                        cell_ui['left'].config(text=f"{quad_data[3]}")
                        cell_ui['center'].config(text=center_text_val, fg=center_fg_color) # Show 'here' if applicable
                        quad_idx +=1

    def set_status_message(self, message): 
        """
        Updates the status message label in the GUI's control panel.
        Used for errors, algorithm status, or completion messages.
        
        Args:
            message (str): The message to display.
        """
        self.output_label.config(text=message)

    def perform_iteration_delay(self, base_delay=0.2):
        """
        Handles GUI updates and introduces a delay, scaled by the speed slider.
        This is used for visualizing algorithm steps at a controllable pace.

        Args:
            base_delay (float): The base delay duration in seconds before speed adjustment.
                                If 0 or None, only `update_idletasks` is called.
        """
        if base_delay and base_delay > 0:
            self.master.update() # Process all pending UI events
            # Calculate actual delay based on speed slider value (higher slider value = faster/shorter delay)
            time.sleep(base_delay / float(self.speed_slider.get()))
        else:
            self.master.update_idletasks() # Process only idle tasks, less forceful than update()

    def _run_generic_algorithm(self, algorithm_class, **kwargs):
        """
        A generic helper method to initiate the execution of an RL algorithm.
        It handles disabling UI controls and managing existing algorithm instances.
        If an algorithm is already running, it stops it before starting the new one,
        using `master.after` to ensure proper event processing by Tkinter.

        Args:
            algorithm_class (class): The class of the RL algorithm to run (e.g., ValueIterationAlgorithm).
            **kwargs: Additional keyword arguments to pass to the algorithm's constructor.
        """
        self._set_controls_enabled(False) # Disable UI controls first
        
        # If an algorithm instance already exists and is marked as running
        if self.current_algorithm_instance and self.current_algorithm_instance.is_running:
            self.current_algorithm_instance.stop() # Stop the currently running algorithm
            # Schedule the actual start of the new algorithm.
            # This `after` delay allows Tkinter to process the stop event and UI updates
            # before the new algorithm potentially blocks or heavily loads the event loop.
            self.master.after(50, lambda: self._actually_run_generic_algorithm(algorithm_class, **kwargs))
            return
        
        # If no algorithm is running, proceed to run the new one directly
        self._actually_run_generic_algorithm(algorithm_class, **kwargs)

    def _actually_run_generic_algorithm(self, algorithm_class, **kwargs):
        """
        Internal helper method that performs the actual instantiation and start of an algorithm.
        This method is called either directly by `_run_generic_algorithm` or after a delay
        if a previous algorithm was being stopped.

        Args:
            algorithm_class (class): The class of the RL algorithm to run.
            **kwargs: Additional keyword arguments for the algorithm's constructor.
        """
        self._set_controls_enabled(False) # Ensure controls are disabled
        
        mdp_model = self._get_mdp_model_from_ui() # Retrieve MDP parameters from UI
        if mdp_model is None:  # If MDP model creation failed (e.g., invalid input)
            self._set_controls_enabled(True) # Re-enable controls
            return # Do not proceed with algorithm execution
        
        # Create an instance of the specified algorithm class
        self.current_algorithm_instance = algorithm_class(mdp_model, self, **kwargs)
        
        # Start the algorithm. Algorithms might have a 'start' (non-blocking, for iterative display)
        # or 'run_algorithm' (potentially blocking, for QL interactive) method.
        if hasattr(self.current_algorithm_instance, 'start') and callable(getattr(self.current_algorithm_instance, 'start')):
            self.current_algorithm_instance.start()
        elif hasattr(self.current_algorithm_instance, 'run_algorithm') and callable(getattr(self.current_algorithm_instance, 'run_algorithm')):
            # This path is typically for algorithms that might have their own loop or specific execution flow.
            self.current_algorithm_instance.run_algorithm() 

    # --- Button Click Handler Methods ---
    def _run_value_iteration_clicked(self): 
        """Handles click event for the 'Run Value Iteration' button."""
        self._run_generic_algorithm(ValueIterationAlgorithm)
        
    def _run_policy_iteration_clicked(self): 
        """Handles click event for the 'Run Policy Iteration' button."""
        self._run_generic_algorithm(PolicyIterationAlgorithm)
        
    def _run_q_learning_clicked(self): 
        """Handles click event for the 'Run Q-Learning' (interactive) button."""
        self._run_generic_algorithm(QLearningAlgorithm)
        
    def _run_epsilon_greedy_clicked(self, decaying=False): 
        """
        Handles click event for the 'Run Epsilon Greedy' and 'Run Decaying E-Greedy' buttons.
        
        Args:
            decaying (bool): True if decaying epsilon is selected, False otherwise.
        """
        self._run_generic_algorithm(EpsilonGreedyAlgorithm, decaying_epsilon=decaying)

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.

    This class provides a common structure and shared functionalities for various RL algorithms,
    such as storing V-values, Q-values, policy, and managing the algorithm's running state.
    It also includes helper methods for preparing display data for the GUI.

    Attributes:
        mdp (mdp.model.MDP): The MDP model instance the algorithm operates on.
        app (GridworldApp): The interface to the main application, used for UI updates.
        V (dict): Stores V-values for states, e.g., {state: value}.
        Q (dict): Stores Q-values for state-action pairs, e.g., {state: {action: value}}.
        policy (dict): Stores the policy, mapping states to actions, e.g., {state: action}.
        is_running (bool): Flag indicating if the algorithm is currently executing.
        iteration (int): Current iteration count for iterative algorithms.
        max_iterations (int): Maximum number of iterations before an algorithm stops automatically.
    """
    def __init__(self, mdp_model, app_interface): 
        """
        Initializes the RLAlgorithm.

        Args:
            mdp_model (mdp.model.MDP): The MDP model.
            app_interface (GridworldApp): The application interface for UI updates.
        """
        self.mdp = mdp_model
        self.app = app_interface # Interface to GridworldApp for UI updates
        
        # Initialize V-values to 0.0 for all algorithmically relevant states (non-walls)
        self.V = {s: 0.0 for s in self.mdp.algo_states} 
        # Initialize Q-values to 0.0 for all state-action pairs in relevant states
        self.Q = {s: {a: 0.0 for a in self.mdp.actions} for s in self.mdp.algo_states} 
        # Initialize policy to None for all relevant states
        self.policy = {s: None for s in self.mdp.algo_states} 
        
        self.is_running = False # Algorithm is not running initially
        self.iteration = 0      # Initialize iteration counter
        self.max_iterations = 2000 # Default maximum iterations

    def start(self): 
        """
        Starts the algorithm's execution. 
        This method should be overridden by subclasses that implement non-blocking,
        step-by-step execution suitable for Tkinter's event loop (e.g., using `master.after`).
        """
        self.is_running = True
        self.iteration = 0
        # Disabling controls is now primarily handled by GridworldApp._run_generic_algorithm
        raise NotImplementedError("Subclasses must implement the start method for asynchronous execution.")
        
    def stop(self):
        """
        Stops the current algorithm execution and re-enables UI controls via the app interface.
        """
        self.is_running = False # Set the flag to stop ongoing iterations
        self.app._set_controls_enabled(True) # Re-enable UI controls

    def _prepare_v_display_tuples(self):
        """
        Prepares V-values and the current policy for display in the V-mode grid.

        Returns:
            list: A list of (v_score_str, direction_str) tuples, ordered for the 9 displayable cells.
                  Returns an empty list if states are not yet defined.
        """
        tuples = []
        # Predefined order of displayable cells (non-wall states)
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        if not self.mdp.states: return tuples # Guard against empty states

        for state_coord in ordered_display_cells:
            v_value = self.V.get(state_coord, 0.0) # Default to 0.0 if state not in V
            policy_action = self.policy.get(state_coord, "") # Default to empty string if no policy
            tuples.append((f"{v_value:.2f}", policy_action)) # Format V-value to 2 decimal places
        return tuples
    
    def _prepare_q_display_quads(self, q_data_source=None):
        """
        Prepares Q-values for display in the Q-mode grid. 
        Each displayable cell will show four Q-values (up, right, down, left).

        Args:
            q_data_source (dict, optional): A dictionary of Q-values to use. 
                                            If None, `self.Q` is used. This allows using
                                            temporarily computed Q-values (e.g., in VI/PI for policy extraction).

        Returns:
            list: A list of (q_up_str, q_right_str, q_down_str, q_left_str) tuples,
                  ordered for the 9 displayable cells. Returns an empty list if states are not defined.
        """
        quads = []
        source_q_values = q_data_source if q_data_source is not None else self.Q
        # Predefined order of displayable cells
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]
        if not self.mdp.states: return quads

        for state_coord in ordered_display_cells:
            q_s_vals = source_q_values.get(state_coord, {}) # Get Q-values for the current state
            # Format each Q-value to 2 decimal places, defaulting to 0.0 if an action is missing
            quads.append((
                f"{q_s_vals.get('up', 0.0):.2f}", 
                f"{q_s_vals.get('right', 0.0):.2f}",
                f"{q_s_vals.get('down', 0.0):.2f}", 
                f"{q_s_vals.get('left', 0.0):.2f}"
            ))
        return quads

    def get_current_display_data_for_toggle(self):
        """
        Prepares a comprehensive set of display data suitable for when the user
        toggles the display mode (V-mode <-> Q-mode).

        - For Value Iteration (VI) and Policy Iteration (PI), which primarily maintain V-values,
          this method computes the corresponding Q-values from V-values and the policy.
        - For Q-Learning (QL) and Epsilon-Greedy (EG), which primarily maintain Q-values,
          this method derives the policy from the Q-values. V-values are also prepared from Q-values (max_a Q(s,a)).

        Returns:
            dict: A dictionary containing data for both V-mode and Q-mode displays,
                  the current agent state (if applicable), and a flag for Q-learning active status.
                  Example: {'v_display_tuples': [...], 'q_display_quads': [...], 
                            'current_agent_state': (r,c), 'q_learn_active': True/False}
        """
        # Prepare V-values and current policy (policy might be updated later for QL/EG)
        # For QL/EG, self.V might not be explicitly maintained by the algorithm, so derive it.
        if isinstance(self, (QLearningAlgorithm, EpsilonGreedyAlgorithm)):
            for s_v in self.mdp.algo_states:
                if s_v in self.Q and self.Q[s_v]:
                    self.V[s_v] = max(self.Q[s_v].values())
                elif s_v in self.mdp.terminal_states: # Ensure terminal states have their fixed values
                    self.V[s_v] = 1.0 if s_v == (0,3) else -1.0 if s_v == (1,3) else 0.0
                else:
                    self.V[s_v] = 0.0 # Default for other states if no Q-values
        v_tuples = self._prepare_v_display_tuples()
        
        q_for_display_dict = {} # To hold Q-values, either from self.Q or derived from V
        
        if isinstance(self, (ValueIterationAlgorithm, PolicyIterationAlgorithm)):
            # For VI/PI, V-values are primary. Derive Q-values for display.
            for s in self.mdp.algo_states: 
                if s in self.mdp.terminal_states: 
                    # For terminal states, Q-values for display can be set to the state's V-value
                    q_for_display_dict[s] = {a: self.V.get(s,0.0) for a in self.mdp.actions} 
                    continue
                action_qs = {}
                for a in self.mdp.actions:
                    expected_val = 0
                    # Calculate sum over s': T(s,a,s') * V(s')
                    if s in self.mdp.transition_model and a in self.mdp.transition_model[s]:
                        for next_s, prob in self.mdp.transition_model[s][a].items():
                            # Wall state (1,1) has V-value 0 and is not in self.V if self.V is from algo_states
                            if next_s != (1,1): expected_val += prob * self.V.get(next_s,0.0) 
                    # Q(s,a) = R(s,a) + gamma * sum_{s'} T(s,a,s')V(s')
                    action_qs[a] = self.mdp.rewards[s][a] + self.mdp.discount_factor * expected_val
                q_for_display_dict[s] = action_qs
            q_display_quads = self._prepare_q_display_quads(q_data_source=q_for_display_dict)
        else: # For QL/EG, Q-values are primary. Derive policy.
            q_display_quads = self._prepare_q_display_quads() # Uses self.Q directly
            # Derive policy from self.Q for display
            for s_pol in self.mdp.algo_states: 
                if s_pol not in self.mdp.terminal_states and self.Q.get(s_pol) and self.Q[s_pol]:
                    # Get action with max Q-value for the state
                    best_action = max(self.Q[s_pol], key=self.Q[s_pol].get, default=None)
                    if best_action: self.policy[s_pol] = best_action
            # Re-prepare V-tuples because policy might have been updated for QL/EG
            v_tuples = self._prepare_v_display_tuples() 

        return {
            'v_display_tuples': v_tuples, 
            'q_display_quads': q_display_quads, 
            'current_agent_state': getattr(self, 'current_state', None), # Agent position if applicable
            'q_learn_active': isinstance(self, (QLearningAlgorithm, EpsilonGreedyAlgorithm))
        }

    def _prepare_flat_display_data_from_v_and_q_dict(self, q_values_dict):
        """
        Prepares both V-value/policy tuples and Q-value quads for display,
        primarily used by Value Iteration and Policy Iteration where Q-values are computed
        as part of the iteration or policy improvement step.

        Args:
            q_values_dict (dict): A dictionary of Q-values {state: {action: value}}
                                  computed during an algorithm's iteration.

        Returns:
            tuple: (v_display_tuples, q_display_quads)
                   - v_display_tuples: List of (v_score_str, direction_str) for V-mode.
                   - q_display_quads: List of (q_up, q_right, q_down, q_left) for Q-mode.
        """
        v_display_tuples = []
        q_display_quads = []
        # Predefined order of displayable cells
        ordered_display_cells = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (2,3)]

        for state_coord in ordered_display_cells:
            # Prepare V-value and policy string from self.V and self.policy
            v_score_str = f"{self.V.get(state_coord, 0.0):.2f}"
            direction_str = self.policy.get(state_coord, "")
            v_display_tuples.append((v_score_str, direction_str))
            
            # Prepare Q-value strings from the provided q_values_dict
            q_s_vals = q_values_dict.get(state_coord, {}) # Get Q-values for the current state
            q_display_quads.append((
                f"{q_s_vals.get('up', 0.0):.2f}", 
                f"{q_s_vals.get('right', 0.0):.2f}",
                f"{q_s_vals.get('down', 0.0):.2f}", 
                f"{q_s_vals.get('left', 0.0):.2f}"
            ))
        return v_display_tuples, q_display_quads

class ValueIterationAlgorithm(RLAlgorithm):
    """
    Implements the Value Iteration algorithm for solving an MDP.
    It iteratively computes the optimal V-values for each state and then derives the policy.
    """
    def __init__(self, mdp_model, app_interface):
        """
        Initializes the ValueIterationAlgorithm.

        Args:
            mdp_model (mdp.model.MDP): The MDP model.
            app_interface (GridworldApp): The application interface for UI updates.
        """
        super().__init__(mdp_model, app_interface)
        # Initialize V-values for terminal states to their known fixed rewards.
        # These values will not be updated during iteration.
        if (0,3) in self.V: self.V[(0,3)] = 1.0  # Positive terminal state
        if (1,3) in self.V: self.V[(1,3)] = -1.0  # Negative terminal state

    def start(self):
        """
        Starts the Value Iteration process.
        The process runs asynchronously with Tkinter's event loop using `master.after`
        to perform iterations step-by-step for visualization.
        """
        self.is_running = True
        self.iteration = 0
        self.app._initialize_grid_display() # Ensure grid is set up for current display mode
        self.app.ql_mode = False # Value Iteration is not Q-learning based for display purposes
        self._perform_one_iteration() # Start the first iteration

    def _perform_one_iteration(self):
        """
        Performs a single iteration of Value Iteration.
        This involves:
        1. Calculating new V-values for all non-terminal states based on the Bellman optimality equation.
        2. Computing the maximum change (delta) in V-values to check for convergence.
        3. Extracting the policy from the updated V-values (or intermediate Q-values).
        4. Refreshing the GUI display.
        5. Scheduling the next iteration if not converged and max iterations not reached.
        """
        # Stop if the algorithm has been externally flagged to stop
        if not self.is_running: 
            self.app._set_controls_enabled(True) # Re-enable UI controls
            return 
        
        # Retrieve epsilon from UI for convergence threshold calculation
        try: 
            epsilon = float(self.app.epsilon_entry.get())
        except ValueError: 
            self.app.set_status_message("Error: Epsilon must be a number.")
            self.stop() # Stop algorithm due to error
            return
        
        # Calculate convergence threshold based on Bellman error bound.
        # Handles discount_factor == 0 or 1 to prevent division by zero or incorrect formula application.
        if self.mdp.discount_factor == 0.0 or self.mdp.discount_factor == 1.0:
            threshold = epsilon
        else:
            threshold = epsilon * (1 - self.mdp.discount_factor) / self.mdp.discount_factor
            
        delta = 0  # Maximum change in V-value in this iteration
        new_V = self.V.copy() # Create a copy to store new V-values for this iteration
        q_values_for_v_iteration = {} # To store Q-values computed during this iteration for policy extraction/display

        # Iterate over all states relevant to the algorithm (non-wall states)
        for s in self.mdp.algo_states:
            if s in self.mdp.terminal_states: continue # Skip terminal states, their V-values are fixed
            
            max_action_value = float('-inf') # Initialize max Q-value for state s
            current_action_q_values = {} # Q-values for all actions from state s
            
            # Calculate Q(s,a) for all actions 'a' from state 's'
            for a in self.mdp.actions:
                action_reward = self.mdp.rewards[s][a] # R(s,a)
                expected_future_value = 0
                # Sum over s': T(s,a,s') * V_k(s')
                if s in self.mdp.transition_model and a in self.mdp.transition_model[s]:
                    for next_s, prob in self.mdp.transition_model[s][a].items():
                        if next_s != (1,1): # Exclude wall state from value calculation
                            expected_future_value += prob * self.V[next_s] # Using V from previous iteration (self.V)
                
                # Q_k+1(s,a) = R(s,a) + gamma * sum_{s'} T(s,a,s')V_k(s')
                current_action_q_values[a] = action_reward + self.mdp.discount_factor * expected_future_value
                max_action_value = max(max_action_value, current_action_q_values[a])
            
            new_V[s] = max_action_value # V_k+1(s) = max_a Q_k+1(s,a)
            q_values_for_v_iteration[s] = current_action_q_values # Store computed Q-values
            delta = max(delta, abs(new_V[s] - self.V[s])) # Update max change for convergence check
        
        self.V = new_V # Update V-values for the next iteration
        
        # Extract policy greedily from the Q-values computed in this iteration
        for s_policy in self.mdp.algo_states:
            if s_policy not in self.mdp.terminal_states and \
               q_values_for_v_iteration.get(s_policy) and q_values_for_v_iteration[s_policy]:
                best_action = max(q_values_for_v_iteration[s_policy], 
                                  key=q_values_for_v_iteration[s_policy].get, default=None)
                if best_action: self.policy[s_policy] = best_action
        
        # Prepare data for GUI update
        v_tuples, q_quads = self._prepare_flat_display_data_from_v_and_q_dict(q_values_for_v_iteration)
        self.app.refresh_display_from_algorithm_data(
            {'v_display_tuples': v_tuples, 'q_display_quads': q_quads, 'q_learn_active': False},
            iteration_delay_info=(False, 0) # No individual step delay, overall delay managed by master.after
        )
        
        self.iteration +=1 # Increment iteration counter
        
        # Check for convergence or max iterations
        if delta > threshold and self.iteration < self.max_iterations:
            # Schedule the next iteration using Tkinter's `after` method for visualization delay
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000) # Base delay 0.2s, adjusted by speed
            self.app.master.after(delay_ms, self._perform_one_iteration)
        else:
            # Algorithm converged or reached max iterations
            status = 'converged' if delta <= threshold else 'stopped (max iterations)'
            self.app.set_status_message(f"Value iteration {status} after {self.iteration} iterations.")
            self.stop() # Stop the algorithm and re-enable UI

class PolicyIterationAlgorithm(RLAlgorithm):
    """
    Implements the Policy Iteration algorithm for solving an MDP.
    It alternates between Policy Evaluation (calculating V-values for the current policy)
    and Policy Improvement (updating the policy greedily based on new V-values).
    """
    def __init__(self, mdp_model, app_interface):
        """
        Initializes the PolicyIterationAlgorithm.

        Args:
            mdp_model (mdp.model.MDP): The MDP model.
            app_interface (GridworldApp): The application interface for UI updates.
        """
        super().__init__(mdp_model, app_interface)
        # Initialize V-values for terminal states
        if (0,3) in self.V: self.V[(0,3)] = 1.0
        if (1,3) in self.V: self.V[(1,3)] = -1.0
        # Initialize a random policy for all non-terminal states
        for s_init in self.mdp.algo_states:
            if s_init not in self.mdp.terminal_states: 
                self.policy[s_init] = random.choice(self.mdp.actions) 

    def start(self):
        """
        Starts the Policy Iteration process.
        Runs asynchronously with Tkinter using `master.after` for step-by-step visualization.
        """
        self.is_running = True
        self.iteration = 0
        self.app._initialize_grid_display() # Set up grid for current display mode
        self.app.ql_mode = False # Not Q-learning based for display
        self._perform_one_policy_iteration_step() # Start the first iteration step

    def _policy_evaluation(self, epsilon_eval): 
        """
        Performs the Policy Evaluation step: iteratively computes V-values for the current policy.
        This is essentially running Value Iteration for a fixed policy until V-values converge.

        Args:
            epsilon_eval (float): The convergence threshold for V-value updates during evaluation.

        Returns:
            dict: The converged V-values (eval_V) for the current policy.
        """
        # Calculate convergence threshold for policy evaluation
        if self.mdp.discount_factor == 0.0 or self.mdp.discount_factor == 1.0:
            threshold_eval = epsilon_eval
        else:
            threshold_eval = epsilon_eval * (1 - self.mdp.discount_factor) / self.mdp.discount_factor
            
        eval_V = self.V.copy() # Start with current V-values (or zeros if first time)
        
        # Iteratively update V-values until convergence
        while True: 
            delta_eval = 0 # Max change in V-value for this evaluation iteration
            new_eval_V = eval_V.copy() # Store new V-values for this sub-iteration
            
            for s_eval in self.mdp.algo_states: # For each non-wall state
                if s_eval in self.mdp.terminal_states: continue # Skip terminal states
                
                old_v_s_eval = eval_V[s_eval] # V_k(s)
                action_to_eval = self.policy[s_eval] # Get action from current policy pi(s)
                
                if action_to_eval is None: # Should not happen with random initialization
                    new_eval_V[s_eval] = 0 
                    continue 
                
                action_reward_eval = self.mdp.rewards[s_eval][action_to_eval] # R(s, pi(s))
                expected_future_val_eval = 0
                # Sum over s': T(s, pi(s), s') * V_k(s')
                if s_eval in self.mdp.transition_model and action_to_eval in self.mdp.transition_model[s_eval]:
                     for next_s_eval, prob_eval in self.mdp.transition_model[s_eval][action_to_eval].items():
                        if next_s_eval != (1,1): # Exclude wall state
                            expected_future_val_eval += prob_eval * eval_V[next_s_eval]
                
                # V_k+1(s) = R(s, pi(s)) + gamma * sum_{s'} T(s, pi(s), s')V_k(s')
                new_eval_V[s_eval] = action_reward_eval + self.mdp.discount_factor * expected_future_val_eval
                delta_eval = max(delta_eval, abs(new_eval_V[s_eval] - old_v_s_eval))
            
            eval_V = new_eval_V # Update V-values for next sub-iteration
            if delta_eval <= threshold_eval: break # Converged
        return eval_V

    def _perform_one_policy_iteration_step(self):
        """
        Performs one step of Policy Iteration, which includes:
        1. Policy Evaluation: Calculate V-values for the current policy.
        2. Policy Improvement: Update the policy greedily based on the new V-values.
        3. Check for policy stability.
        4. Refresh GUI and schedule the next step if policy is not stable.
        """
        if not self.is_running: 
            self.app._set_controls_enabled(True)
            return
        
        # Get epsilon for policy evaluation convergence
        try: 
            epsilon = float(self.app.epsilon_entry.get())
        except ValueError: 
            self.app.set_status_message("Error: Epsilon must be a number.")
            self.stop()
            return

        # --- Policy Evaluation ---
        self.V = self._policy_evaluation(epsilon) 
        
        # --- Policy Improvement ---
        policy_stable = True # Flag to check if the policy changed in this iteration
        q_for_improvement = {} # To store Q-values computed for policy improvement
        
        for s_improve in self.mdp.algo_states: # For each non-wall state
            if s_improve in self.mdp.terminal_states: 
                # For terminal states, Q-values can be set to the state's V-value for display consistency
                q_for_improvement[s_improve] = {a:self.V.get(s_improve,0.0) for a in self.mdp.actions}
                continue
            
            old_action_improve = self.policy[s_improve] # Current action under policy pi_k(s)
            action_q_s_improve = {} # Q-values for all actions from s_improve, using V from evaluation
            
            # Calculate Q(s,a) for all actions 'a' using the evaluated V-values (self.V)
            for a_improve in self.mdp.actions:
                action_reward_improve = self.mdp.rewards[s_improve][a_improve] # R(s,a)
                expected_future_val_improve = 0
                # Sum over s': T(s,a,s') * V_pi(s')
                if s_improve in self.mdp.transition_model and a_improve in self.mdp.transition_model[s_improve]:
                    for next_s_improve, prob_improve in self.mdp.transition_model[s_improve][a_improve].items():
                        if next_s_improve != (1,1): # Exclude wall
                            expected_future_val_improve += prob_improve * self.V[next_s_improve]
                action_q_s_improve[a_improve] = action_reward_improve + self.mdp.discount_factor * expected_future_val_improve
            
            q_for_improvement[s_improve] = action_q_s_improve # Store computed Q-values
            
            if action_q_s_improve: # Ensure there are Q-values to choose from
                # pi_k+1(s) = argmax_a Q(s,a)
                best_action_improve = max(action_q_s_improve, key=action_q_s_improve.get, default=None)
                if best_action_improve: self.policy[s_improve] = best_action_improve
                if best_action_improve != old_action_improve: 
                    policy_stable = False # Policy changed for this state
        
        # Prepare data for GUI update
        v_tuples_pi, q_quads_pi = self._prepare_flat_display_data_from_v_and_q_dict(q_for_improvement)
        self.app.refresh_display_from_algorithm_data(
            {'v_display_tuples': v_tuples_pi, 'q_display_quads': q_quads_pi, 'q_learn_active': False},
            iteration_delay_info=(False, 0) # No individual step delay
        )
        
        self.iteration += 1
        
        # Check for policy stability or max iterations
        if not policy_stable and self.iteration < self.max_iterations :
            delay_ms = int((0.2 / float(self.app.speed_slider.get())) * 1000) # Base delay 0.2s
            self.app.master.after(delay_ms, self._perform_one_policy_iteration_step)
        else:
            status = 'converged' if policy_stable else 'stopped (max iterations)'
            self.app.set_status_message(f"Policy iteration {status} after {self.iteration} iterations.")
            self.stop()
    
class QLearningAlgorithm(RLAlgorithm):
    """
    Implements user-interactive Q-Learning.
    The agent's moves are determined by user key presses. The algorithm updates
    Q-values based on these interactions.
    """
    def __init__(self, mdp_model, app_interface):
        """
        Initializes the QLearningAlgorithm.

        Args:
            mdp_model (mdp.model.MDP): The MDP model.
            app_interface (GridworldApp): The application interface for UI updates.
        """
        super().__init__(mdp_model, app_interface)
        # N_sa: Counts visits to state-action pairs, used for decaying alpha (learning rate)
        self.N_sa = {s: {a: 0 for a in self.mdp.actions} for s in self.mdp.algo_states} 
        self.current_state = (2,0) # Agent's starting position in the gridworld
        self.move_var = tk.StringVar() # Tkinter variable to capture user's chosen move (from key press)
        
        # Initialize Q-values for terminal states to 0. Actions from terminal states are not meaningful.
        for terminal_s in self.mdp.terminal_states: 
            if terminal_s in self.Q: # Ensure state exists in Q table keys
                for a_term in self.mdp.actions: 
                    self.Q[terminal_s][a_term] = 0.0 

    def start(self):
        """
        Starts the Q-learning algorithm. For interactive Q-learning, this typically
        means setting up for user input, so it calls `run_algorithm` which contains the main loop.
        """
        self.run_algorithm() # The main logic is in run_algorithm for interactive QL

    def _on_key(self, event):
        """
        Handles key press events (Up, Down, Left, Right) for agent movement.
        Updates `self.move_var`, which `run_algorithm` waits on.

        Args:
            event (tk.Event): The key press event.
        """
        if event.keysym in ['Up', 'Down', 'Left', 'Right']: 
            self.move_var.set(event.keysym.lower()) # Store the chosen action (e.g., "up")

    def run_algorithm(self): 
        """
        Runs the interactive Q-learning algorithm.
        This method contains a loop that waits for user key presses (actions),
        updates Q-values based on the chosen action and resulting state/reward,
        and refreshes the GUI. The loop continues for a fixed number of moves.
        This method is blocking in the sense that it uses `master.wait_variable`.
        """
        self.is_running = True 
        # Set UI modes for Q-learning display
        self.app.ql_mode = True 
        self.app.current_grid_mode = 'q'
        self.app._initialize_q_grid_display() # Initialize grid for Q-values
        
        # Get initial alpha (learning rate) from UI
        try: 
            initial_alpha = float(self.app.a_value_entry.get())
        except ValueError: 
            self.app.set_status_message("Error: Alpha(QL) must be a number.")
            self.stop()
            return
        
        move_count = 0
        max_moves = 40 # Limit for interactive session
        
        # Bind arrow key presses to the _on_key method
        self.app.master.bind('<Key>', self._on_key)
        self.app.set_status_message("Use the arrow keys to move the marker.")
        # Initial display refresh to show agent at start position
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))

        # Main interactive loop
        while move_count < max_moves:
            # Wait for user to press an arrow key (which sets self.move_var)
            self.app.master.wait_variable(self.move_var)
            action = self.move_var.get() # Get the action chosen by user
            self.move_var.set("") # Reset for next key press
            
            if not self.is_running: break # Check if algorithm was stopped externally
            if not action or action not in self.mdp.actions : continue # Ignore invalid actions
            
            # Simulate action: Determine next state (s_next) based on transition model
            s_next = self.current_state # Default if no transition defined (should not happen in valid MDP)
            if self.current_state in self.mdp.transition_model and \
               action in self.mdp.transition_model[self.current_state]:
                probs = self.mdp.transition_model[self.current_state][action]
                next_states_list, probabilities_list = list(probs.keys()), list(probs.values())
                if next_states_list: # Ensure there are possible next states
                    s_next = random.choices(next_states_list, weights=probabilities_list, k=1)[0]
            
            # Determine reward: R(s,a) from MDP, overridden if s_next is terminal
            r_val = self.mdp.rewards[self.current_state][action] 
            if s_next == (0,3): r_val = 1.0  # Reward for reaching positive terminal state
            elif s_next == (1,3): r_val = -1.0 # Reward for reaching negative terminal state
            
            # Q-value update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
            # This is only done if the current state is not terminal.
            if self.current_state not in self.mdp.terminal_states:
                max_next_q_val = 0.0
                # Find max_a' Q(s',a')
                if s_next in self.Q and s_next not in self.mdp.terminal_states: 
                    max_next_q_val = max(self.Q[s_next].values()) if self.Q[s_next] else 0.0
                
                target_q_val = r_val + self.mdp.discount_factor * max_next_q_val # TD Target: r + gamma * max_a' Q(s',a')
                
                self.N_sa[self.current_state][action] += 1 # Increment visit count for (s,a)
                # Decaying alpha: alpha_sa = initial_alpha / (1 + N(s,a))
                alpha = initial_alpha / (1 + self.N_sa[self.current_state][action]) 
                
                # Perform the Q-value update
                self.Q[self.current_state][action] = \
                    (1-alpha) * self.Q[self.current_state][action] + alpha * target_q_val
            
            self.current_state = s_next # Move to the next state
            # Refresh GUI display with updated Q-values and agent position
            self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(True, 0.1)) # Short delay after move
            
            move_count += 1
            
            # If agent reaches a terminal state, reset its position after a short delay
            if self.current_state in self.mdp.terminal_states:
                self.app.perform_iteration_delay(0.5) # Pause to show terminal state reached
                self.current_state = (2,0) # Reset to start state
                # Refresh display to show agent at new start position
                self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        
        # Loop finished (max_moves reached or stopped)
        self.app.set_status_message(f"Q-Learning stopped after {move_count} moves.")
        self.app.master.unbind('<Key>') # Remove key binding
        self.stop() # Ensure algorithm is fully stopped and UI enabled

class EpsilonGreedyAlgorithm(QLearningAlgorithm): 
    """
    Implements Epsilon-Greedy Q-Learning with automated step-by-step execution.
    This algorithm explores randomly with probability epsilon and exploits (chooses the best
    known action) with probability 1-epsilon. It can also use a decaying epsilon.
    Inherits from QLearningAlgorithm for Q-value update logic and state management.
    """
    def __init__(self, mdp_model, app_interface, decaying_epsilon=False):
        """
        Initializes the EpsilonGreedyAlgorithm.

        Args:
            mdp_model (mdp.model.MDP): The MDP model.
            app_interface (GridworldApp): The application interface for UI updates.
            decaying_epsilon (bool): If True, epsilon decays over time. Otherwise, it's fixed.
        """
        super().__init__(mdp_model, app_interface) # Call parent QLearningAlgorithm constructor
        self.decaying_epsilon = decaying_epsilon
        self.max_moves = 300 # More moves for automated exploration/exploitation
        self.move_count = 0 

    def start(self):
        """
        Starts the Epsilon-Greedy Q-Learning process.
        Runs asynchronously using Tkinter's `master.after` for visualization.
        """
        self.is_running = True
        self.move_count = 0
        # Set UI modes for Q-learning display
        self.app.ql_mode = True
        self.app.current_grid_mode = 'q'
        self.app._initialize_q_grid_display() 
        self.app.set_status_message("Running Epsilon-Greedy...")
        # Initial display update (e.g., agent at start)
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        self._perform_one_eg_step() # Start the first automated step

    def _perform_one_eg_step(self):
        """
        Performs a single step of Epsilon-Greedy Q-Learning.
        This includes:
        1. Choosing an action (epsilon-greedy).
        2. Simulating the action and observing the next state and reward.
        3. Updating the Q-value for the taken state-action pair.
        4. Handling epsilon decay if enabled.
        5. Refreshing the GUI and scheduling the next step.
        """
        # Stop if algorithm flagged or max moves reached
        if not self.is_running or self.move_count >= self.max_moves:
            if self.is_running : self.app.set_status_message(f"Epsilon-Greedy finished after {self.move_count} moves.")
            self.stop()
            return

        # Get alpha and epsilon from UI (allows dynamic changes, though less common for automated runs)
        try: 
            initial_alpha = float(self.app.a_value_entry.get())
            epsilon_val = float(self.app.epsilon_entry.get())
        except ValueError:
            self.app.set_status_message("Error: Alpha or Epsilon is not a valid number.")
            self.stop()
            return

        # Epsilon-greedy action selection
        if random.random() < epsilon_val: # Explore: choose a random action
            action_chosen = random.choice(self.mdp.actions)
        else: # Exploit: choose the action with the highest Q-value for the current state
            # Fallback to random choice if no Q-values yet for current_state or state not in Q
            action_chosen = max(self.Q[self.current_state], key=self.Q[self.current_state].get) \
                            if self.current_state in self.Q and self.Q[self.current_state] \
                            else random.choice(self.mdp.actions)
        
        # Simulate action and get next state (s_next)
        s_next = self.current_state 
        if self.current_state in self.mdp.transition_model and \
           action_chosen in self.mdp.transition_model[self.current_state]:
            probs_eg = self.mdp.transition_model[self.current_state][action_chosen]
            next_states_list_eg, probabilities_list_eg = list(probs_eg.keys()), list(probs_eg.values())
            if next_states_list_eg: 
                s_next = random.choices(next_states_list_eg, weights=probabilities_list_eg, k=1)[0]
        
        # Determine reward
        r_val_eg = self.mdp.rewards[self.current_state][action_chosen]
        if s_next == (0,3): r_val_eg = 1.0
        elif s_next == (1,3): r_val_eg = -1.0
        
        # Q-value update (same logic as in QLearningAlgorithm)
        if self.current_state not in self.mdp.terminal_states:
            max_next_q_val_eg = 0.0
            if s_next in self.Q and s_next not in self.mdp.terminal_states:
                max_next_q_val_eg = max(self.Q[s_next].values()) if self.Q[s_next] else 0.0
            
            target_q_val_eg = r_val_eg + self.mdp.discount_factor * max_next_q_val_eg
            self.N_sa[self.current_state][action_chosen] += 1
            alpha_eg = initial_alpha / (1 + self.N_sa[self.current_state][action_chosen])
            self.Q[self.current_state][action_chosen] = \
                (1-alpha_eg) * self.Q[self.current_state][action_chosen] + alpha_eg * target_q_val_eg
        
        self.current_state = s_next # Move to next state
        # Refresh GUI (no individual step delay here, overall pace by master.after)
        self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0))
        
        self.move_count += 1
        
        # Handle decaying epsilon if enabled
        if self.decaying_epsilon:
            epsilon_val = max(0.001, epsilon_val * 0.99)  # Decay epsilon, with a floor of 0.001
            self.app.epsilon_entry.delete(0, tk.END) # Update UI with new epsilon
            self.app.epsilon_entry.insert(0, str(f"{epsilon_val:.4f}"))
        
        # Schedule the next step
        delay_ms_eg = int((0.05 / float(self.app.speed_slider.get())) * 1000) # Base delay 0.05s for faster auto steps
        next_step_delay_after = delay_ms_eg # Default delay for next step
        
        # If agent reaches a terminal state, reset position and add a longer pause
        if self.current_state in self.mdp.terminal_states:
            self.current_state = (2,0) # Reset to start state
            # Refresh display to show agent at new start position
            self.app.refresh_display_from_algorithm_data(self.get_current_display_data_for_toggle(), iteration_delay_info=(False,0)) 
            next_step_delay_after = 500 # Longer pause (0.5s) after hitting terminal state
            
        self.app.master.after(next_step_delay_after, self._perform_one_eg_step)
    
def main():
    """Main function to create and run the Tkinter application."""
    root = tk.Tk()
    app = GridworldApp(root) 
    root.mainloop() 

if __name__ == "__main__":
    main()

