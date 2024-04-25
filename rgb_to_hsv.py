import numpy as np

import cv2
import argparse

'''
Configure program args
'''
def configureArgs():    
    global args 

    # configure arguments
    # -h/--help is automatically generated
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--red", type=int, required=True, help="red value [0,255]")
    ap.add_argument("-g", "--green", type=int, required=True, help="green value [0,255]")
    ap.add_argument("-b", "--blue", type=int, required=True, help="blue value [0,255]")
    args = vars(ap.parse_args())

if __name__ == "__main__":
    configureArgs()

    r = args["red"]
    g = args["green"]
    b = args["blue"]
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)

    print("HSV value: [{}, {}, {}]".format(hsv[0][0][0], hsv[0][0][1], hsv[0][0][2]))
