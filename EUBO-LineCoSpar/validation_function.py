"""
This file contains specific functions used during the validation phase of the experiment in the optimization.py file.

- validation_test(best_translate, challenger, i):
Evaluate the consistency of user preference between his prefered point and a random challenger.
"""
import random
import time

from core_functions import translate_param
def validation_test(best_translate, challenger, i):
    """
    Performs a validation test by asking the user to compare the final selected configuration (best_translate)
    with a challenger configuration. The order of presentation is randomized.

    Args:
        best_translate (list or tensor): The configuration found by the optimization process.
        challenger (tensor): A randomly selected configuration to compare against the best.
        i (int): The current validation iteration number.

    Returns:
        int: User's preference input.
             0 = No preference
             1 = Preference for option 1
             2 = Preference for option 2
    """
    # Randomly decide whether best_translate or challenger appears first
    bit = random.randint(0, 1)
    
    print('\nValidation', i)
    
    if bit == 1:
        # Present best_translate first, then challenger
        print(f"Send point 1: {best_translate}")
        time.sleep(3)
        print(f"Send point 2: {translate_param(challenger)}")
        time.sleep(2)
    else:
        # Present challenger first, then best_translate
        print(f"Send point 1: {translate_param(challenger)}")
        time.sleep(3)
        print(f"Send point 2: {best_translate}")
        time.sleep(2)
    
    # Ask the user for their preference
    while True:
        choice = input("Preference? (0 = None, 1 = Option 1 or 2 = Option 2): ")
        if choice in ['0', '1', '2']:
            return int(choice)  # Return user's valid input
        print("Enter 0, 1 or 2.")  # Prompt again for valid input
