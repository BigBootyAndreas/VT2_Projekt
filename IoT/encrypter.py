
import fernetfile

#
# Encrypt
#

def encrypt(source, destination, key):

    with open(source, 'rb') as fin, fernetfile.open(destination, mode='wb', fernet_key=key) as fout:
        while True:
            data = fin.read(7777)
            if not data:
                break
            fout.write(data)
    
    print("Encrypted CSV file")
