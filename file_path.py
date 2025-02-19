import os

#Defining the path for the onedrive file
def path(person):
    if person == "Ali" or person == "ali":
        directory = r"C:\\Users\\Aljun\\Aalborg Universitet\\P8 - testfile_dont_delete\\"
        return directory
    elif person == "Sofus" or person == "Sofus":
        directory = ""
        return directory
    elif person == "Theis" or person == "theis":
        directory = ""
        return directory
    elif person == "Jacob" or person == "jacob":
        directory = ""
        return directory
    elif person == "Tobias" or person == "tobias":
        directory = ""
        return directory
    elif person == "Viktor" or person == "viktor":
        directory = ""
        return directory
    else:
        print(f"{person} doesn't exist")
        person = input("Who's running the script(Enter your name with a capital letter): ")
        path(person)
