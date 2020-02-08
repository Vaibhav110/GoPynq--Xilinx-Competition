import cv2
import numpy as np
import socket
import Con_array_wa
import time
from pynq import Overlay
from pynq import Xlnk

no_train=96	# Number of training smaples=96
k_top=90
counter=0
port = 12345	# Port defined for IoT connection

s= socket.socket()	
s.bind(('',port))
s.listen(5)
mean = Con_array_wa.mean

eigen_vector_val = Con_array_wa.eigen_vector
numpy_vector_value = np.reshape(eigen_vector_val,(5184,k_top)) # For a 72 X 72 image
eigen_vec = np.transpose(numpy_vector_value)
        
eigen_coff_val = Con_array_wa.eigen_coff
numpy_coff_value = np.reshape(eigen_coff_val,(no_train,k_top))
eigen_coff = np.transpose(numpy_coff_value) 

c, addr = s.accept()	# Connection for IoT
print ('Got connection from', addr ) # Connection established


cap = cv2.VideoCapture(0)	# Webcam Initialzation
i=0
while(True):
        ret, frame = cap.read()	# Webacm Frame defined
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	# capture image
     
################################################################################### Sharpening Image

        overlay = Overlay('/home/xilinx/jupyter_notebooks/Project/Bitstream/Convolve_aryan/design_1.bit') # Define Overlay for sharpening captured image
	convolve_ip = overlay.convolve_0
	#print(overlay.ip_dict.keys())
        #print(convolve_ip.register_map)
	xlnk = Xlnk()
	A_buffer = xlnk.cma_array(shape=(9,), dtype=np.ubyte)	# Mapping inputs for processing
        B_buffer = xlnk.cma_array(shape=(1,), dtype=np.ubyte)
	A_dram_addr = A_buffer.physical_address
        B_dram_addr = B_buffer.physical_address
	A_reg_offset = convolve_ip.register_map.A.address
        B_reg_offset = convolve_ip.register_map.B.address
        CTRL_reg_offset = convolve_ip.register_map.CTRL.address

	img=np.array(img, dtype=np.int)
	img1=np.pad(img, ((1,1),(1,1)), 'constant')
	m=img.shape[0]
        n=img.shape[1]
	for i in range(m):
    		for j in range(n):
			A_buffer= img1[i:i+3,j:j+3].flatten('F')

			convolve_ip.write(A_reg_offset,A_dram_addr)	# Sending inputs
			convolve_ip.write(B_reg_offset,B_dram_addr)	# Getting output
			convolve_ip.write(CTRL_reg_offset,1)

			while(not(convolve_ip.register_map.CTRL.AP_DONE)):
			img(i,j)=B_buffer

################################################################################### Segmentation

        th=5*float(img[int(m/2),10])/218
        for i in range(n):
            if(int(img[int(m/2),i-1])-int(img[int(m/2),i])>th and i>10):
                break;
        j=n-1
        while(int(img[int(m/2),j])-int(img[int(m/2),j-1])<th):
            j-=1
        w=j-i+20
        y=int((m-w)/2)
        x=i
        img=img[y-10:y+w,x-10:x+w+10]
	img=cv2.resize(img, (72,72), interpolation = cv2.INTER_AREA)
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,90,			# Circle detection
                            param1=100,param2=35,minRadius=10,maxRadius=72)

	circles = np.uint16(np.around(circles))
	mask = np.zeros(img.shape,dtype=np.uint8)
	for i in circles[0,:]:
	  if((i[0]+i[2]<img.shape[1])&(i[0]>i[2])&(i[1]+i[2]<img.shape[0])&(i[1]+5>i[2])):
	     cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1,8,0)
	mask=mask/255
	print(img.shape)
	for i in range(72):
	    for j in range(72):
        	if(mask[i,j]==0):
        	    img[i,j]=255
        #cv2.imwrite('crop.jpg',img)
        img1=np.array(img, dtype=np.float)	# Final extracted/cropped emoji
        img2=img1/255
        img3=img2.flatten('F')		# Flatten cropped image for pca technique
        img4=img3-mean

        
################################################################################### Matrix multiplication
        overlay_2 = Overlay('/home/xilinx/jupyter_notebooks/Project/Bitstream/Convolve_aryan/mat_mult.bit')	# Define Overlay for matrix multiplication 
	mul_ip = overlay_2.mat_multiply_0
        mul_ip.read(0x0c)
	xlnk = Xlnk()

        A_buffer_2 = xlnk.cma_array(shape=(5184,), dtype=np.float32)	# Mapping inputs for processing
        B_buffer_2 = xlnk.cma_array(shape=(5184,), dtype=np.float32)
        C_buffer_2 = xlnk.cma_array(shape=(1,), dtype=np.float32)
	A_dram_addr = A_buffer_2.physical_address
        B_dram_addr = B_buffer_2.physical_address
        C_dram_addr = C_buffer_2.physical_address
	A_reg_offset = mul_ip.register_map.A.address
        B_reg_offset = mul_ip.register_map.B.address
        C_reg_offset = mul_ip.register_map.C.address
        CTRL_reg_offset = mul_ip.register_map.CTRL.address
        
	B_buffer_2=img4		# Temporary variable to store image
	for i in range(k_top):
	    A_buffer_2=eigen_vec[i]
	    mul_ip.write(A_reg_offset,A_dram_addr)	# Sending Inputs (A - Eigen matrix)
            mul_ip.write(B_reg_offset,B_dram_addr)	# Sending Inputs (B - Image)
            mul_ip.write(C_reg_offset,C_dram_addr)	# Getting Output (C)
	    mul_ip.write(CTRL_reg_offset,1)
            while(not(mul_ip.register_map.CTRL.AP_DONE)):
            coff_test[i]=C_buffer_2[0]

#####################################################################################	# PCA technique detection
        
        coff_ext_test=np.tile(coff_test,(no_train,1) )
        coff_ext_test=coff_ext_test.transpose()
	diff_sq=np.square(eigen_coff - coff_ext_test)
        error= np.sum(diff_sq, axis=0)
	result = np.where(error == np.amin(error))
        print(result[0][0])
        res = int(result[0][0])
        counter=counter+1
        array=[0,0,0,0,0,0]

        if (res <=15 ):
            print ("1")
            print("concerned")
            array[0]=array[0]+1;
        elif (res <= 31 ):
            print ("2")
            print("cry")
            array[1]=array[1]+1;
        elif (res <=47 ):
            print ("3")
            print("glasses")
            array[2]=array[2]+1;
        elif (res <=63 ):
            print ("4")
            print("laugh")
            array[3]=array[3]+1;
        elif (res <=79 ):
            print ("5")
            print("smile")
            array[4]=array[4]+1;
        elif (res <=95 ):
            print ("6")
            print("unwell")
            array[5]=array[5]+1;
        print("end")
        
        if(counter==10):
            counter=0
            maxi=array.index(max(array))
	    array=[0,0,0,0,0,0]
################################################################################### Send signal via IoT depending on detected emoji
            if(maxi==0): 
                c.send(b'0')	
            if(maxi==1): 
                c.send(b'1')
            if(maxi==2): 
                c.send(b'2')
            if(maxi==3): 
                c.send(b'3')
            if(maxi==4): 
                c.send(b'4')
            if(maxi==5): 
                c.send(b'5')
            cv2.imwrite('frame1.jpg',frame)
            cv2.imwrite('image1.jpg',img)
            time.sleep(5)
	    k=cv2.waitKey(1) & 0xFF 
            if(k==ord('q')):
                s.close()
                c.close()
                break
            




