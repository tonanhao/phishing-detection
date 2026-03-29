# Step 1: Clone project
!git clone https://github.com/YOUR_USERNAME/phishing-detection.git
%cd phishing-detection

# Step 2: Upload CSV file
from google.colab import files
print("Upload PhiUSIIL_Phishing_URL_Dataset.csv:")
uploaded = files.upload()

# Step 3: Install dependencies
!pip install -r requirements.txt

# Step 4: Run training
!python main.py --dataset-mode phiussiil --epochs 20 --batch-size 128

# Step 5: Download model
files.download("experiments/best_model_phiussiil.pt")
