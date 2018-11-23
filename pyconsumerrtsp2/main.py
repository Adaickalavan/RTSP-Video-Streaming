from dotenv import load_dotenv
import kafkapc_python as pc
import os
import cv2
import message
import dataprocessing.alg as alg

def main():
    # Create consumer 
    consumer = pc.Consumer(
        os.getenv('TOPICNAME'),
        os.getenv('KAFKAPORT'),
        os.getenv('CONSUMERGROUP')
        )

    # Prepare openCV window
    print(cv2.__version__)
    windowName = "RTSPvideo2"
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 240, 160)

    #Instantiate a signal processing model
    model = alg.Model()

    # Start consuming video
    for m in consumer:
        #Read message contents
        val = m.value
        print("Time:",m.timestamp,", Topic:",m.topic) 

        #Message handler
        img = message.handler(val)

        #Show image
        cv2.imshow(windowName, img)
        cv2.waitKey(1)

        #Process the message
        model.run(img)
        
    consumer.close()                                    
    
    return

if __name__ == "__main__":
    # Load environment variable
    load_dotenv(override=True)
    # Call main function
    main()
