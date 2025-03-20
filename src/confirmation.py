# Confirmation function
def ask_for_confirmation():
    while True:
        user_input = input("Is the segmentation acceptable? (yes/no): ").strip().lower()
        if user_input in ['yes', 'y']:
            return True
        elif user_input in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please type 'yes' or 'no'.")