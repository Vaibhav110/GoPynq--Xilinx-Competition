# Import socket module 
import socket  
import winsound  
import struct            
  
# Create a socket object 
s = socket.socket()          
  
# Define the port on which you want to connect 
port = 1234                
prev_rec_data=0
# connect to the server on local computer 
s.connect(('10.107.88.120', port)) 

# receive data from the server 
while True:
    rec_data= int(s.recv(1)) 
    print(rec_data)

    #Play the specific required Songs	
    if(rec_data!=prev_rec_data):
        if(rec_data==0):
            print('concerned')
            winsound.PlaySound('07 - Tera Yaar Hoon Main (SongsMp3.Com).wav',winsound.SND_FILENAME)
        elif(rec_data==1):
            print('cry')
            winsound.PlaySound('ro_mat.wav',winsound.SND_FILENAME)
        elif(rec_data==2):
            print('SWAG- Glasses')
            winsound.PlaySound('Imagine Dragons - Thunder (192  kbps) (ovh).wav',winsound.SND_FILENAME)
        elif(rec_data==3):
            print('Laugh')
            winsound.PlaySound('Pharrell Williams - Happy (Official Music Video).wav',winsound.SND_FILENAME)
        elif(rec_data==4):
            print('Smile')
            winsound.PlaySound('Sadda Haq (Rockstar).wav',winsound.SND_FILENAME)
        elif(rec_data==5):
            print('Unwell')
            winsound.PlaySound('Taylor_Swift_Ft_Dixie_Chicks_-_Soon_You_ll_Get_Better_Emvidowealth.wav',winsound.SND_FILENAME)
     
    prev_rec_data=rec_data

# close the connection 
s.close() 
