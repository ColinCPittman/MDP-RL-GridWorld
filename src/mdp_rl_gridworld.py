#!/usr/bin/env python
# coding: utf-8

# # Imports and Gloabls

# In[1]:


import tkinter as tk
import time
import random
cells = {}

# global variables for MDP
states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]

# global variables for GUI elements
current_grid_mode = "v"
ql_mode = False
cells = {}
root = None
grid_frame = None
control_panel_frame = None
value_iteration_button = None
q_learning_button = None
policy_iteration_button = None
epsilon_greedy_q_button = None
reset_button = None
x_value_entry = None
r_value_entry = None
epsilon_entry = None
discount_entry = None
speed_slider = None


# # MDP Functions

# In[2]:


def get_mdp_model(): #this could be decomposed further since I have a section for MDP funtions, for now I will keep it to make the value iteration function easier to follow in code
    states = [
        (0, 0), (0, 1), (0, 2), (0, 3),  # row 0, top row
        (1, 0), (1, 1), (1, 2), (1, 3),  # Row 1, including wall at (1,1) which needs to be accounted for in generating tuples for printout
        (2, 0), (2, 1), (2, 2), (2, 3)   # Row 2
    ]
    terminal_states = [(0, 3), (1, 3)]

    actions = ["up", "down", "left", "right"]
    
    # rewards, (following example here: https://www.youtube.com/watch?v=UuTkioxL9bQ), this is a dictionary of {state: {action: reward}}
    reward_step_cost = float(r_value_entry.get()) # get living reward cost
    rewards = {}
    for state in states: 
        if state in terminal_states:
            rewards[state] = {action: 0 for action in actions} # no reward for actions in terminal states, to simplify
        elif state == (1, 1): # wall state
            rewards[state] = {action: 0 for action in actions} # no reward for actions in wall state
        else:
            rewards[state] = {action: reward_step_cost for action in actions}
    
    # handling the terminal state rewards directly with overwrite
    rewards[(0, 3)] = {action: 1.0 for action in actions} # +1 
    rewards[(1, 3)] = {action: -1.0 for action in actions} # -1 
    
    
    # transition model generation setup, again as a dictionary of {state: {action: {next_state: probability}}}
    x_prob_value = float(x_value_entry.get())
    prob_intended = x_prob_value / 100.0
    prob_side = (100.0 - x_prob_value) / 200.0

    # rows and column differences will be use to check logic on moves. boundaries are cheecked to see if it exceeding grid rows or column or the wall at (1,1)
    action_index_difference = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    side_actions = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["up", "down"],
        "right": ["up", "down"],
    }
    
    transition = {}
    grid_rows = 3 
    grid_cols = 4
    
    for state in states: 
        transition[state] = {}
        if state in terminal_states or state == (1, 1): # no transitions from terminal or wall states
            for action in actions:
                transition[state][action] = {state: 1.0} # stay in the same state with prob 1.0
            continue 
    
        for action in actions:
            transition[state][action] = {} # initialize next state probabilities for this action
    
            # intended next state
            row, col = state
            row_diff, col_diff = action_index_difference[action]
            intended_next_state_candidate = (row + row_diff, col + col_diff)
    
            # handle boundaries and wall for intended state
            if not (0 <= intended_next_state_candidate[0] < grid_rows and 0 <= intended_next_state_candidate[1] < grid_cols) or intended_next_state_candidate == (1, 1):
                intended_next_state = state # stay in current state if intended move is invalid
            else:
                intended_next_state = intended_next_state_candidate
    
    
            transition[state][action][intended_next_state] = prob_intended # add intended next state and probability
    
            # side move next states and probabilities
            for side_action in side_actions[action]:
                side_row_diff, side_col_diff = action_index_difference[side_action]
                side_next_state_candidate = (row + side_row_diff, col + side_col_diff)
    
                # boundaries and wall for side states
                if not (0 <= side_next_state_candidate[0] < grid_rows and 0 <= side_next_state_candidate[1] < grid_cols) or side_next_state_candidate == (1, 1):
                    side_next_state = state # stay in place if move is invalid
                else:
                    side_next_state = side_next_state_candidate
                transition[state][action][side_next_state] = transition[state][action].get(side_next_state, 0.0) + prob_side # add side move and probability
    discount = float(discount_entry.get())
    return states, rewards, transition, discount, actions


# # GUI Functions

# In[3]:


def setup_gui():
    global root, grid_frame, control_panel_frame, value_iteration_button, q_learning_button, policy_iteration_button, epsilon_greedy_q_button, reset_button
    global output_label, x_value_entry, r_value_entry, epsilon_entry, discount_entry, speed_slider, display_button, a_value_entry
    root = tk.Tk()
    root.title("Gridworld Display")

    # frame for the grid
    grid_frame = tk.Frame(root)
    grid_frame.grid(row=0, column=0, sticky="nsew")
 
    # frame for the panel of controls at the bottom
    control_panel_frame = tk.Frame(root)
    control_panel_frame.grid(row=1, column=0, sticky="ew")

    #  buttons (row 0 of control_panel_frame)
    value_iteration_button = tk.Button(control_panel_frame, text="Run Value Iteration", command=value_iteration)
    value_iteration_button.grid(row=0, column=0, padx=5, pady=5)

    q_learning_button = tk.Button(control_panel_frame, text="Run Q-Learning", command=q_learning)
    q_learning_button.grid(row=0, column=2, padx=5, pady=5)

    policy_iteration_button = tk.Button(control_panel_frame, text="Run Policy Iteration", command=policy_iteration)
    policy_iteration_button.grid(row=0, column=1, padx=5, pady=5)

    epsilon_greedy_q_button = tk.Button(control_panel_frame, text="Run Epsilon Greedy", command=epsilon_greedy)
    epsilon_greedy_q_button.grid(row=0, column=3, padx=5, pady=5)
    
    #replaced the reset button with a button that calls epsilon_greedy with a decaying epsilon option. The reset button was just confusing things anyway since it didn't properly pause execution of previously running algoirthms. Plus, the run buttons reset everything anyway. 
    reset_button = tk.Button(control_panel_frame, text="Run Decaying E-Greedy", command=lambda: epsilon_greedy(decaying = True)) 
    reset_button.grid(row=0, column=4, padx=5, pady=5)
    
    display_button = tk.Button(control_panel_frame, text="Cycle Display Mode", command=toggle_display_mode)
    display_button.grid(row=0, column=5, padx=5, pady=5)
    
    # input boxes and labels (row 1 of control_panel_frame) 
    x_value_label = tk.Label(control_panel_frame, text="X Value:")
    x_value_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    x_value_entry = tk.Entry(control_panel_frame, width=5)
    x_value_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    x_value_entry.insert(0, "90") # default X value

    r_value_label = tk.Label(control_panel_frame, text="R Value:")
    r_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="e")
    r_value_entry = tk.Entry(control_panel_frame, width=5)
    r_value_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
    r_value_entry.insert(0, "-0.04") # default R value

    a_value_label = tk.Label(control_panel_frame, text="Alpha(QL):")
    a_value_label.grid(row=1, column=4, padx=5, pady=5, sticky="e")
    a_value_entry = tk.Entry(control_panel_frame, width=5)
    a_value_entry.grid(row=1, column=5, padx=5, pady=5, sticky="w")
    a_value_entry.insert(0, "0.5") # default alpha value

    epsilon_label = tk.Label(control_panel_frame, text="Epsilon:")
    epsilon_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
    epsilon_entry = tk.Entry(control_panel_frame, width=5)
    epsilon_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
    epsilon_entry.insert(0, "0.001") # default epsilon value
    
    discount_label = tk.Label(control_panel_frame, text="Discount:")
    discount_label.grid(row=2, column=2, padx=5, pady=5, sticky="e")
    discount_entry = tk.Entry(control_panel_frame, width=5)
    discount_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")
    discount_entry.insert(0, "0.99") # default discount value

    output_label = tk.Label(control_panel_frame, text="", width=50, anchor="w")
    output_label.grid(row=3, column=6, columnspan=6, padx=5, pady=5, sticky="w")

    speed_slider_label = tk.Label(control_panel_frame, text="Speed Multiplier:")
    speed_slider_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
    speed_slider = tk.Scale(control_panel_frame, from_=.5, to=1.5, orient=tk.HORIZONTAL, resolution=0.01)
    speed_slider.set(1)
    speed_slider.grid(row=3, column=1, padx=5, pady=5, sticky="w")
    
    # root window row and column weights, again for resizing
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)


# In[4]:


def initialize_q_grid():
    global current_grid_mode, cells, ql_mode
    output_label.config(text="")

    # setting their values to zero if they exist, attmept to optimize GUI refreses by removing cells.clear() from the q_display_grid
    if cells:
        for key in cells:
            if isinstance(cells[key], dict):
                cells[key]['top'].config(text="0.00")
                cells[key]['right'].config(text="0.00")
                cells[key]['bottom'].config(text="0.00")
                cells[key]['left'].config(text="0.00")
                if (0, 3) in cells and isinstance(cells[(0, 3)], dict):
                    if not ql_mode:
                        cells[(0, 3)]['center'].config(text="1.00", font=("Comfortaa", 12))
                    else:
                        cells[(0, 3)]['center'].config(text="", font=("Comfortaa", 12))
                    cells[(0, 3)]['top'].config(text="")
                    cells[(0, 3)]['right'].config(text="")
                    cells[(0, 3)]['bottom'].config(text="")
                    cells[(0, 3)]['left'].config(text="")
                
            
                if (1, 3) in cells and isinstance(cells[(1, 3)], dict):
                    if not ql_mode:
                        cells[(1, 3)]['center'].config(text="-1.00", font=("Comfortaa", 12))
                    else:
                        cells[(1, 3)]['center'].config(text="", font=("Comfortaa", 12))
                    cells[(1, 3)]['top'].config(text="")
                    cells[(1, 3)]['right'].config(text="")
                    cells[(1, 3)]['bottom'].config(text="")
                    cells[(1, 3)]['left'].config(text="")
                # wall cell
                if (1, 1) in cells and isinstance(cells[(1, 1)], dict):
                    cells[(1, 1)]['frame'].config(bg="grey")
                    cells[(1, 1)]['center'].config(text="", bg="grey")
                    cells[(1, 1)]['top'].config(text="", bg="grey")
                    cells[(1, 1)]['right'].config(text="", bg="grey")
                    cells[(1, 1)]['bottom'].config(text="", bg="grey")
                    cells[(1, 1)]['left'].config(text="", bg="grey")
    else:
        initial_q_quadtuples = []
        for _ in range(9):
            initial_q_quadtuples.append(("0.00", "0.00", "0.00", "0.00"))
        q_display_grid(grid_frame, initial_q_quadtuples)
        


# In[5]:


def initialize_v_grid():
    global current_grid_mode
    output_label.config(text="")
    current_grid_mode = "v"
    initial_v_tuples = []
    for _ in range(9):
        initial_v_tuples.append((0.00, "up")) # initializing to "Up" as default direction, following the examples in the slides
    v_display_grid(grid_frame, initial_v_tuples)


# In[6]:


def initialize_grid():
    global current_grid_mode
    if current_grid_mode == "v":
        initialize_v_grid()
    else:
        initialize_q_grid()


# In[7]:


def display_grid(grid_frame, tuples_list, type = None):
    global current_grid_mode
    if current_grid_mode == "v":
        v_display_grid(grid_frame, tuples_list, type = None)
    else:
        q_display_grid(grid_frame, tuples_list)


# In[8]:


def toggle_display_mode():
    global current_grid_mode
    
    for widget in grid_frame.winfo_children():
        widget.destroy() # tearing down configuration for previous mode
    
    cells.clear()
    
    if current_grid_mode == "v":
        current_grid_mode = "q"
        initialize_q_grid()
    else:
        current_grid_mode = "v"
        initialize_v_grid()


# **Display function for V-Score board**
# - This is the method to call when updating the display for the board which contains only v-scores and directions.
# - Takes in a list of tuples (v_score, direction), for each cell.
# - Tuples information is populated into cells starting from the top left and ending with the bottom right.

# In[9]:


def v_display_grid(grid_frame, tuples_list, type=None):
    if len(tuples_list) != 9:
        raise ValueError("tuples_list must contain exactly 9 tuples.")

    global cells
    cells.clear() 


    for i in range(3):
        grid_frame.grid_rowconfigure(i, weight=1, minsize=100) 
    for j in range(4):
        grid_frame.grid_columnconfigure(j, weight=1, minsize=100)

    tuple_index = 0
    for row in range(3):
        for col in range(4):
            is_terminal_positive = row == 0 and col == 3
            is_terminal_negative = row == 1 and col == 3
            is_wall = row == 1 and col == 1
            
            if is_terminal_positive:
                text = "1.00"
            elif is_terminal_negative:
                text = "-1.00"
            elif is_wall:
                text = ""
            else:
                if type is None:
                    text = f"Max Reward:\n\n{tuples_list[tuple_index][0]} if going {tuples_list[tuple_index][1]}."
                else:
                    text = tuples_list[tuple_index][1]
                tuple_index += 1

            cell_key = (row, col)
            cell = tk.Label(grid_frame, text=text, relief=tk.SOLID, padx=10, pady=5, 
                           width=10, height=5, font=("Comfortaa", 12))
            cell.grid(row=row, column=col, sticky="nsew")
            
            
            cell.config(width=10, height=5)
            
            if is_wall:
                cell.config(bg="grey")
                
            cells[cell_key] = cell


# **Display function for Q-Score board**
# - This is the fuction that is called to display the board which contains the q-scores and of the 4 directions in each cell.
# - This takes in a quadtuple of q_scores, which are used to populate each cell.
#     - The quadtuples populate the up, right, down, and left directions respectively when read left-to-right
# - The cells of the board are populated beginning with the top-left cell and ending with the bottom-right cell. 

# In[10]:


def q_display_grid(grid_frame, quadtuple_list, q_learn = False):
    global cells
    display_states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
    if len(quadtuple_list) != 9:
        raise ValueError("quadtuple_list must contain exactly 9 quadtuples.")
    
    
    for i in range(3):
        grid_frame.grid_rowconfigure(i, weight=1, minsize=100)  
    for j in range(4):
        grid_frame.grid_columnconfigure(j, weight=1, minsize=100)  
    
    
    for row in range(3):
        for col in range(4):
            pos = (row, col)
            # skip creating cell frames that already exist
            if pos in cells and isinstance(cells[pos], dict):
                continue
                

            cell_frame = tk.Frame(grid_frame, relief=tk.SOLID, bd=1, width=100, height=100)
            cell_frame.grid(row=row, column=col, sticky="nsew")
            cell_frame.grid_propagate(False) 
            
        
            for i in range(3):
                cell_frame.grid_rowconfigure(i, weight=1)
            for j in range(3):
                cell_frame.grid_columnconfigure(j, weight=1)
            
            center_label = tk.Label(cell_frame, text="", font=("Comfortaa", 10))
            center_label.grid(row=1, column=1)
            
            top_label = tk.Label(cell_frame, text="0.00", anchor="s")
            top_label.grid(row=0, column=1, sticky="ew")
            
            right_label = tk.Label(cell_frame, text="0.00", anchor="w")
            right_label.grid(row=1, column=2, sticky="ns")
            
            bottom_label = tk.Label(cell_frame, text="0.00", anchor="n")
            bottom_label.grid(row=2, column=1, sticky="ew")
            
            left_label = tk.Label(cell_frame, text="0.00", anchor="e")
            left_label.grid(row=1, column=0, sticky="ns")
            
            cells[pos] = {
                'frame': cell_frame,
                'center': center_label,
                'top': top_label,
                'right': right_label,
                'bottom': bottom_label,
                'left': left_label
            }
    
    # updating existing cells and manually configuring the terminal and wall states
    tuple_index = 0
    for pos in display_states:
        row, col = pos
        quad = quadtuple_list[tuple_index]
        
        if pos in cells and isinstance(cells[pos], dict):
            cell_labels = cells[pos]
            cell_labels['top'].config(text=f"{quad[0]}")
            cell_labels['right'].config(text=f"{quad[1]}")
            cell_labels['bottom'].config(text=f"{quad[2]}")
            cell_labels['left'].config(text=f"{quad[3]}")
            cell_labels['center'].config(text="")
            
        
        tuple_index += 1
    

    if (0, 3) in cells and isinstance(cells[(0, 3)], dict):
        if not q_learn:
            cells[(0, 3)]['center'].config(text="1.00", font=("Comfortaa", 12))
        cells[(0, 3)]['top'].config(text="")
        cells[(0, 3)]['right'].config(text="")
        cells[(0, 3)]['bottom'].config(text="")
        cells[(0, 3)]['left'].config(text="")
    

    if (1, 3) in cells and isinstance(cells[(1, 3)], dict):
        if not q_learn:
            cells[(1, 3)]['center'].config(text="-1.00", font=("Comfortaa", 12))
        cells[(1, 3)]['top'].config(text="")
        cells[(1, 3)]['right'].config(text="")
        cells[(1, 3)]['bottom'].config(text="")
        cells[(1, 3)]['left'].config(text="")
    # wall cell
    if (1, 1) in cells and isinstance(cells[(1, 1)], dict):
        cells[(1, 1)]['frame'].config(bg="grey")
        cells[(1, 1)]['center'].config(text="", bg="grey")
        cells[(1, 1)]['top'].config(text="", bg="grey")
        cells[(1, 1)]['right'].config(text="", bg="grey")
        cells[(1, 1)]['bottom'].config(text="", bg="grey")
        cells[(1, 1)]['left'].config(text="", bg="grey")


# # Part 1 - Value Iteration

# In[11]:


def value_iteration():
    initialize_grid()
    states, rewards, transition, discount, actions = get_mdp_model()
    epsilon = float(epsilon_entry.get())
    threshold = epsilon * (1 - discount) / discount #definine the breakout condition here to avoid performing the calculation on every loop
    global ql_mode
    ql_mode = False
    # initialize values ignoring the wall
    V = {s: 0 for s in states if s != (1, 1)}
    # manually setting the terminal states to their rewards
    V[(0, 3)] = rewards[(0, 3)]["up"]  # should be 1.0
    V[(1, 3)] = rewards[(1, 3)]["up"]  # should be -1.0
    
    iteration = 0
    while True:
        iteration += 1
        delta = 0 #reset delta to zero each iteration
        new_V = V.copy()

        # calculate the v-scores for each 
        for s in states:
            if s == (1, 1) or s in [(0, 3), (1, 3)]:
                continue  # skip wall and terminal states
                
            # calculate value for each action and take the max
            max_value = float('-inf')
            for a in actions:
                value = rewards[s][a]
                for next_state, prob in transition[s][a].items(): #items is needed to iterate through the key-value pairs of a dictionary.
                    if next_state != (1, 1):  # skip the wall state
                        value += discount * prob * V[next_state]
                max_value = max(max_value, value)
            new_V[s] = max_value
            delta = max(delta, abs(new_V[s] - V[s]))
        # usually the function would return the V at this point but we're going to use it here to determine the optimal policy
        # but our example shows arrows indicating direction 
        V = new_V
        
        # policy determination given final V scores
        policy = {}
        q_values_for_display = {}  # dictionary to store q-values for each state for the q grid mode
        
        for s in states:
            if s == (1, 1) or s in [(0, 3), (1, 3)]:
                policy[s] = None
                continue
     
            action_values = {}
            for a in actions:
                value = rewards[s][a]
                for next_state, prob in transition[s][a].items():
                    if next_state != (1, 1):
                        value += discount * prob * V[next_state]
                action_values[a] = value
            q_values_for_display[s] = action_values
            
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            
        # generate display tuples based on current display mode
        display_states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
        display_tuples = []
        
        global current_grid_mode
        for state in display_states:
            if current_grid_mode == "v":
                v_score_str = f"{V[state]:.2f}"
                direction_str = policy.get(state, "")
                display_tuples.append((v_score_str, direction_str))
            else:  # q-mode
                q_scores = q_values_for_display.get(state, {})
                if q_scores:
                    quadtuple = (
                        f"{q_scores.get('up', 0):.2f}",
                        f"{q_scores.get('right', 0):.2f}",
                        f"{q_scores.get('down', 0):.2f}",
                        f"{q_scores.get('left', 0):.2f}"
                    )
                    display_tuples.append(quadtuple)
        display_grid(grid_frame, display_tuples)
        grid_frame.update()
        wait_time = 0.2 / float(speed_slider.get())
        time.sleep(wait_time)
        
        # check for convergence
        if delta <= threshold:
            output_label.config(text=f"Value iteration converged after {iteration} iterations.")
            break


# # Part 2: Policy Iteration

# In[12]:


def policy_evaluation(policy, discount, epsilon, V):
    initialize_grid() 
    states, rewards, transition, _, actions = get_mdp_model() #removed discount because it was overriding the parameter
    threshold = epsilon * (1 - discount) / discount
    # compute the new V scores for each state with provided policy
    while True: 
        delta = 0
        new_V = V.copy() 
        q_values_for_display = {}
        for s in states:
            if s == (1, 1) or s in [(0, 3), (1, 3)]:
                continue
            old_v = V[s]
            value = 0
            action_to_evaluate = policy[s] 
            action_values = {}
            if action_to_evaluate is not None:
                a = action_to_evaluate
                value_for_action = 0
                for next_state, prob in transition[s][a].items():
                    if next_state != (1, 1):
                        value_for_action += prob * (rewards[s][a] + discount * V[next_state])
                action_values[action_to_evaluate] = value_for_action
                value = value_for_action
            q_values_for_display[s] = action_values
            new_V[s] = value
            delta = max(delta, abs(new_V[s] - old_v))
            
        V = new_V
        if delta <= threshold: 
            break
    return V, q_values_for_display


def policy_iteration():
    initialize_grid()
    global ql_mode
    ql_mode = False
    states, rewards, transition, discount, actions = get_mdp_model() 
    epsilon = float(epsilon_entry.get()) 
    threshold = epsilon * (1 - discount) / discount

    V = {s: 0 for s in states if s != (1, 1)} 
    V[(0, 3)] = rewards[(0, 3)]["up"]
    V[(1, 3)] = rewards[(1, 3)]["up"]

    # setting all policies to up for all valid states
    policy = {s: None for s in states}
    for s in states:
        if s != (1, 1) and s not in [(0, 3), (1, 3)]:
            policy[s] = "up"


    iteration = 0 
    while True: 
        iteration += 1 
        V, q_values_for_display= policy_evaluation(policy, discount, epsilon, V) 

        # improvement
        policy_stable = True 
        for s in states: 
            if s == (1, 1) or s in [(0, 3), (1, 3)]:
                continue
            
            old_action = policy[s] 
            action_values = {} 
            for a in actions: 
                value = rewards[s][a]
                for next_state, prob in transition[s][a].items():
                    if next_state != (1, 1):
                        value += discount * prob * V[next_state]
                action_values[a] = value 

            best_action = max(action_values, key=action_values.get) # argmax
            policy[s] = best_action

            if policy[s] != old_action: 
                policy_stable = False

        # display and wait (visualization part, can keep or adjust)
        display_states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
        display_tuples = []
        global current_grid_mode
        for state in display_states: 
            if current_grid_mode == "v":
                v_score_str = f"{V[state]:.2f}"
                direction_str = policy.get(state, "")
                display_tuples.append((v_score_str, direction_str))
            else:  # q-mode
                q_scores = q_values_for_display.get(state, {})
                if q_scores:
                    quadtuple = (
                        f"{q_scores.get('up', 0):.2f}",
                        f"{q_scores.get('right', 0):.2f}",
                        f"{q_scores.get('down', 0):.2f}",
                        f"{q_scores.get('left', 0):.2f}"
                    )
                    display_tuples.append(quadtuple)
        display_grid(grid_frame, tuple(display_tuples))
        grid_frame.update() 
        global speed_slider 
        wait_time = 0.2 / float(speed_slider.get())
        time.sleep(wait_time)

        if policy_stable:
            output_label.config(text=f"Policy iteration converged after {iteration} iterations.")
            break 


# # Part 3: Q-Learning

# In[13]:


def q_to_quadtuples(Q):
    # convert Q dictionary to a list of quadtuples for easier display of each state
    display_states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
    quad_list = []
    for s in display_states:
        if s in Q:
            quad = (
                f"{Q[s].get('up', 0):.2f}",
                f"{Q[s].get('right', 0):.2f}",
                f"{Q[s].get('down', 0):.2f}",
                f"{Q[s].get('left', 0):.2f}"
            )
        else:
            quad = ("0.00", "0.00", "0.00", "0.00")
        quad_list.append(quad)
    return quad_list
def q_learning():
    global ql_mode
    ql_mode = True
    global cells, root, current_grid_mode, output_label, grid_frame, speed_slider
    
    current_grid_mode = 'q'
    initialize_q_grid()
    grid_frame.update()
    
    states, rewards, transition, discount, actions = get_mdp_model() # this is could be model-free but since I have already built a model I am chosing to use it here.
    initial_alpha = float(a_value_entry.get())
    
    # initialize q-values
    Q = {s: {a: 0.0 for a in actions} for s in states if s != (1, 1)}
    
    # q-values for terminal states, these will be updated during learning
    for a in actions:
        if (0, 3) in Q:
            Q[(0, 3)][a] = 0.0  # starting at 0, will learn toward 1.0
        if (1, 3) in Q:
            Q[(1, 3)][a] = 0.0  # starting at 0, will learn toward -1.0
     # N_sa (frequency table for decaying alpha)
        N_sa = {s: {a: 0 for a in actions} for s in states if s != (1, 1)}
        for a in actions:
            if (0, 3) in N_sa:
                N_sa[(0, 3)][a] = 0
            if (1, 3) in N_sa:
                N_sa[(1, 3)][a] = 0
    current_state = (0, 0)
    move_count = 0
    max_moves = 40

    # to ensure the Q-grid is displayed with the "here" marker
    quad_list = q_to_quadtuples(Q)
    q_display_grid(grid_frame, quad_list, q_learn = True)

    #these next few lines are for a listener for the arrow keys to capture inputs, a common tkinter appraoch I found online
    move_var = tk.StringVar()
    
    def on_key(event): #tk key handler
        if event.keysym in ['Up', 'Down', 'Left', 'Right']:
            move_var.set(event.keysym.lower())
    root.bind('<Key>', on_key)

    def update_here_marker(): # clear and update 'here' marker location, rather than having parameters, we're using the current_state variable
        for pos in cells:
            if isinstance(cells[pos], dict) and pos not in [(0, 3), (1, 3)]:
                cells[pos]['center'].config(text="")
        if current_state in cells and isinstance(cells[current_state], dict):
            cells[current_state]['center'].config(text="here", font=("Comfortaa", 12))

                
    update_here_marker()
    grid_frame.update()
    output_label.config(text="Use the arrow keys to move the marker.")
    while move_count < max_moves:
        root.wait_variable(move_var) # this is like the get action 
        action = move_var.get()
        move_var.set("")
        if action not in actions:
            continue

        # get next state based on transition probabilities
        probs = transition[current_state][action] # separate the key value pair of the state and probability from the actions for use with Random's choice() to generate next move
        next_states = list(probs.keys())
        probabilities = list(probs.values())
        s_next = random.choices(next_states, weights=probabilities, k=1)[0] 
        
        # reward based on the next state we end up in
        if s_next == (0, 3):
            r = 1.0  
        elif s_next == (1, 3):
            r = -1.0  
        else:
            r = rewards[current_state][action]  
        
        # calculate the R(s, a, s′) + γ*max_a′Q(s′, a′) part
        if s_next in [(0, 3), (1, 3)]:
            reward_plus_discounted_best_future_Q = r # ask professor about this, not sure if step cost should be incurred going into terminal states or not
        else:
            max_next = max(Q[s_next].values()) if s_next in Q else 0.0
            reward_plus_discounted_best_future_Q = r + discount * max_next
        
        N_sa[current_state][action] += 1

        # decaying alpha based on visit count, figure 23.8 shows alpha being a function of a frequencies of state action pairs, am dividing it here because I assume this is to encourage novel exploration. Ask the professor about this
        alpha = initial_alpha / (1 + N_sa[current_state][action])
        current_estimated_Q = Q[current_state][action]
        
        Q[current_state][action] = (1-alpha)*current_estimated_Q + alpha * (reward_plus_discounted_best_future_Q)
    
        # display and state update logic
        quad_list = q_to_quadtuples(Q)
        q_display_grid(grid_frame, quad_list, q_learn = True)
    
        current_state = s_next
        update_here_marker()
        grid_frame.update()
        time.sleep(0.1 / float(speed_slider.get()))
        move_count += 1
        
        # restart if terminal state reached
        if current_state in [(0, 3), (1, 3)]:
            if current_state == (0, 3) and (0, 3) in cells and isinstance(cells[(0, 3)], dict):
                for a in actions:
                    Q[(0, 3)][a] = (Q[(0, 3)][a] + r) / 2 # running average of the values found in the terminal states, need to ask the professora about this too, not sure what to do with the terminal state's calculation.
                cells[(0, 3)]['center'].config(text=f"{Q[(0, 3)]['up']:.2f}", font=("Comfortaa", 12))

            elif current_state == (1, 3) and (1, 3) in cells and isinstance(cells[(1, 3)], dict):
                for a in actions:
                    Q[(1, 3)][a] = (Q[(1, 3)][a] + r) / 2
                cells[(1, 3)]['center'].config(text=f"{Q[(1, 3)]['up']:.2f}", font=("Comfortaa", 12))
            
            grid_frame.update()
            time.sleep(0.5)
            current_state = (0, 0)
            update_here_marker()
            grid_frame.update()

    output_label.config(text=f"Q-Learning stopped after {move_count} moves.")
    root.unbind('<Key>')


# # Part 4: Greedy

# In[14]:


def epsilon_greedy(decaying = False):
    global ql_mode
    ql_mode = True
    global cells, root, current_grid_mode, output_label, grid_frame, speed_slider
    
    current_grid_mode = 'q'
    initialize_q_grid()
    grid_frame.update()
    
    states, rewards, transition, discount, actions = get_mdp_model()
    initial_alpha = float(a_value_entry.get())
    epsilon = float(epsilon_entry.get())  # Get epsilon value from UI
    
    # initialize q-values
    Q = {s: {a: 0.0 for a in actions} for s in states if s != (1, 1)}
    
    # q-values for terminal states
    for a in actions:
        if (0, 3) in Q:
            Q[(0, 3)][a] = 0.0
        if (1, 3) in Q:
            Q[(1, 3)][a] = 0.0
    
    # frequency table of states for decaying alpha
    N_sa = {s: {a: 0 for a in actions} for s in states if s != (1, 1)}
    for a in actions:
        if (0, 3) in N_sa:
            N_sa[(0, 3)][a] = 0
        if (1, 3) in N_sa:
            N_sa[(1, 3)][a] = 0
    
    current_state = (0, 0)
    move_count = 0
    max_moves = 100 
    
    # to ensure the Q-grid is displayed with the "here" marker
    quad_list = q_to_quadtuples(Q)
    q_display_grid(grid_frame, quad_list, q_learn=True)
    
    def update_here_marker():
        for pos in cells:
            if isinstance(cells[pos], dict) and pos not in [(0, 3), (1, 3)]:
                cells[pos]['center'].config(text="")
        if current_state in cells and isinstance(cells[current_state], dict):
            cells[current_state]['center'].config(text="here", font=("Comfortaa", 12))
    
    update_here_marker()
    grid_frame.update()
    
    # epsilon-greedy loop
    while move_count < max_moves:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[current_state], key=Q[current_state].get)
        
        # getting next state based on transition probabilities
        probs = transition[current_state][action]
        next_states = list(probs.keys())
        probabilities = list(probs.values())
        s_next = random.choices(next_states, weights=probabilities, k=1)[0]
        
        # setting reward based on the next state we end up in, may need to update depending on how conversation with the professor goes, same question above in q-learning regarding if entering terminal state should also incure a step cost 
        if s_next == (0, 3):
            r = 1.0
        elif s_next == (1, 3):
            r = -1.0
        else:
            r = rewards[current_state][action]
        
        # calculate the R(s, a, s′) + γ*max_a′Q(s′, a′) part of formula 23.7
        if s_next in [(0, 3), (1, 3)]:
            reward_plus_discounted_best_future_Q = r
        else:
            max_next = max(Q[s_next].values()) if s_next in Q else 0.0
            reward_plus_discounted_best_future_Q = r + discount * max_next
        
        N_sa[current_state][action] += 1
        
        # decaying alpha based on visit count, may need to update depending on how conversation with professor goes
        alpha = initial_alpha / (1 + N_sa[current_state][action])
        current_estimated_Q = Q[current_state][action]
        
        Q[current_state][action] = (1-alpha)*current_estimated_Q + alpha * (reward_plus_discounted_best_future_Q)
        
        # GUI display update logic next
        quad_list = q_to_quadtuples(Q)
        q_display_grid(grid_frame, quad_list, q_learn=True)
        
        current_state = s_next
        update_here_marker()
        grid_frame.update()
        time.sleep(0.1 / float(speed_slider.get()))
        move_count += 1
        
        if decaying:
            epsilon = max(0.1, epsilon * 0.99)  
            epsilon_entry.delete(0, tk.END) 
            epsilon_entry.insert(0, str(epsilon))
        
        # reset position if terminal state reached
        if current_state in [(0, 3), (1, 3)]:
            if current_state == (0, 3) and (0, 3) in cells and isinstance(cells[(0, 3)], dict):
                for a in actions:
                    Q[(0, 3)][a] = (Q[(0, 3)][a] + r) / 2 # again, need to ask professor about this, still not sure about these terminal updates
                cells[(0, 3)]['center'].config(text=f"{Q[(0, 3)]['up']:.2f}", font=("Comfortaa", 12))
            
            elif current_state == (1, 3) and (1, 3) in cells and isinstance(cells[(1, 3)], dict):
                for a in actions:
                    Q[(1, 3)][a] = (Q[(1, 3)][a] + r) / 2
                cells[(1, 3)]['center'].config(text=f"{Q[(1, 3)]['up']:.2f}", font=("Comfortaa", 12))
            
            grid_frame.update()
            time.sleep(0.5)
            current_state = (0, 0)
            update_here_marker()
            grid_frame.update()
    
    output_label.config(text=f"Epsilon-Greedy finished after {move_count} moves.")


# # Main Controller

# In[15]:


def main():
    setup_gui() 

    initialize_v_grid()
    
    root.mainloop() 

if __name__ == "__main__":
    main()


# In[ ]:




