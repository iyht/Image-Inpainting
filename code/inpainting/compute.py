## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    sizeof_patch = (2*psiHatP.radius()+1)**2

    confidenceImage_patch, _ = copyutils.getWindow(confidenceImage, psiHatP._coords, psiHatP.radius())

    filled_patch = psiHatP.filled()

    unfilled_confidence_area = np.multiply(confidenceImage_patch.astype(float), filled_patch/255)

    # Replace this dummy value with your own code

    C = np.sum(unfilled_confidence_area)/sizeof_patch/255


    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    inpainted_patch = psiHatP.pixels()
    filled_patch = psiHatP.filled()
    inpainted_patch_grey = cv.cvtColor(inpainted_patch, cv.COLOR_BGR2GRAY)
    #ignore filled pixels
    new_inpainted_patch_grey = np.multiply(inpainted_patch_grey, filled_patch/255)
    sobelx = cv.Sobel(new_inpainted_patch_grey,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(new_inpainted_patch_grey,cv.CV_64F,0,1,ksize=3)
    #calculate gradient
    #absolute make it faster than square root
    magnitude_gradient = np.absolute(sobelx)+np.absolute(sobely)
    #find max
    index = np.unravel_index(np.argmax(magnitude_gradient), magnitude_gradient.shape)#tuple

    
    # Replace these dummy values with your own code
    Dy = sobely[index[0]][index[1]]
    Dx = sobelx[index[0]][index[1]]
    #########################################
    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    inpainted_patch = psiHatP.pixels()
    filled_patch = psiHatP.filled()
    if np.count_nonzero(filled_patch) == 1:
        Ny = None
        Nx = None
        return Ny, Nx
    inpainted_patch_grey = cv.cvtColor(inpainted_patch, cv.COLOR_BGR2GRAY)
    new_inpainted_patch_grey = np.multiply(inpainted_patch_grey, filled_patch/255)
    sobelx = cv.Sobel(new_inpainted_patch_grey,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(new_inpainted_patch_grey,cv.CV_64F,0,1,ksize=3)



    #Replace these dummy values with your own code

    Ny = -1*sobely[psiHatP.radius()][psiHatP.radius()]
    Nx = sobelx[psiHatP.radius()][psiHatP.radius()]


    #########################################

    return Ny, Nx