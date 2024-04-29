import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names=["Peter parker" , "Rebecca Miller "]
# these are the names used for the authentication
usernames=["pparker","rmiller"]
# passwords=["abc123","def456"]
passwords=["XXX","XXX"]

# to convert plane text passwords to hash passowords
hashed_passwords=stauth.Hasher(passwords).generate()

# storing hash passwords into pickle file
file_path=Path(__file__).parent/"hashed_pw.pkl"
#opening up the file and writing them in binary mode
with file_path.open("wb") as file:
    # dumping all the passowrds in the file
    pickle.dump(hashed_passwords,file)

