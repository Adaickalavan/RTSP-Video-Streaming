#Add data processing steps here
#This file is currently filled with dummy sample code

class Model:
    def __init__(self):
        self.avePix = 128 #Dummy initial value

    def run(self, img):
        # Compute and print the average pixel value
        self.avePix = img.mean()
        print("Average pixel value is:", self.avePix)
