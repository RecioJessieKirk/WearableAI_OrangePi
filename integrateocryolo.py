import subprocess

while True:
    user_input = input("\nEnter mode ('yolo' for object detection, 'ocr' for text recognition, 'exit' to quit): ").strip().lower()
    
    if user_input == "yolo":
        print("Starting YOLO Object Detection...")
        process = subprocess.run(["python", "computervision.py"])
        
        # If YOLO exits with sys.exit(1), restart OCR
        if process.returncode == 1:
            print("Switching to OCR Mode...")
            subprocess.run(["python", "eastocrbasiccodecv.py"])
    
    elif user_input == "ocr":
        print("Starting OCR Text Recognition...")
        subprocess.run(["python", "eastocrbasiccodecv.py"])
    
    elif user_input == "exit":
        print("Exiting program.")
        break
    
    else:
        print("Invalid input. Please enter 'yolo', 'ocr', or 'exit'.")
