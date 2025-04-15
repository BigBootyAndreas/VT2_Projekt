import os

#this script works to predefine a path automatically without user input


# Get the user profile path
user_profile = os.environ['USERPROFILE']

# Extract the username from the user profile path
username = user_profile.split('\\')[-1]    

# Now you have the username stored in the variable 'username'

#Pre defying directory path
dir= f"C:\\Users\\{username}\\Aalborg Universitet\\P8 - Product development\\Data\\"
dir2=f"C:\\Users\\{username}\\Aalborg Universitet\\P8 - Projekt\\Product development\\Data\\"
