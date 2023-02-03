import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd="D:\Tesseract-OCR\tesseract.exe"
image =cv2.imread('aaaaa.jpg')
image =imutils.resize(image,width=500)

#we will display original image when it will start finding
cv2.imshow("Original Image",image) #here Original image is the name 
cv2.waitKey(0)
#convert image to gray for reducing dimension and reducing complexity and for working of algorithms
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray scale image",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("smoother image",gray)
cv2.waitKey(0)

#now we will find the edges of image
edged=cv2.Canny(gray,170,200)
cv2.imshow("Canny edge",edged)
cv2.waitKey(0)

#now we will find the contours based on images
cnts,new=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#new is heirarchy-relationship
#RETR_LIST- it retrives all the contours but doesn't create any parent-child relationship
#CHAIN_APPROX_SIMPLE= it removes all the reduntant points and compress the contour by saving memory


#contour-edge image
#draw original image for all its contours
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("Canny after contoursing",image1)
cv2.waitKey(0)


#now we don't want all the contours we are interested only in number plate
#but can't directly locate that so we will sort the on the basis of their areas
#we will select those area which are maximum so will select upto 30 areas
#sorted list of min to maximum
#reverse sorting
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
NumberPlateCount=None
#because currently we don't have any contour or you can say it will show how many number plates are their in image
#to draw top 30 contours we will copy of original image and use
image2=image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
cv2.imshow("Top 30 contours",image2)
cv2.waitKey(0)

#now we will run a fo rloop on our contours to find the best possible contours fo four expectes number plate
count=0
name=1 #name of our cropped image

for i in cnts:
    perimeter=cv2.arcLength(i,True)
    #perimeter is size called as arclength we can find directly in python 
    approx=cv2.approxPolyDP(i,0.02*perimeter,True)
    #approxplydp : approximates the curve of polygon with precision

    if(len(approx)==4): #4 corners
        NumberPlateCount=approx
        x,y,w,h=cv2.boundingRect(i)
        crp_img=image[y:y+h,x:x+w]

        cv2.imwrite(str(name)+'.png',crp_img)
        name+=1
        break
cv2.drawContours(image,[NumberPlateCount],-1,(0,255,0),3)
cv2.imshow("final image",image)
cv2.waitKey(0)

#final image
crop_img_loc='1.png'
cv2.imshow("cropped image",cv2.imread(crop_img_loc))
cv2.waitKey(0)
text=pytesseract.image_to_string(crop_img_loc)
print('number is: ',text)
cv2.waitKey(0)


