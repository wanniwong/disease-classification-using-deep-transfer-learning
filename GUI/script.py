from flask import Flask, flash, request, redirect, url_for, render_template
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import os
from itertools import repeat

app=Flask(__name__)

UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.secret_key = 'GI-disease-prediction'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    global model
    model = models.resnet50(pretrained=True) # define pretrained resnet50 model
    model.fc = nn.Linear(model.fc.in_features, 3) # change final layer to 3 outputs
    pretrained_weights = torch.load('resnet50.pt', map_location='cpu') # get the weights from trained model
    model.load_state_dict(pretrained_weights) # load pretrained weights
    model.eval() # switch to model to eval mode

def get_image(img_path):
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    img = Image.open(img_path)
    return tfms(img).unsqueeze(0)

def get_prediction(img_path):
    img_tensor = get_image(img_path)
    get_model()

    outputs = model(img_tensor)
    _, y_hat = outputs.max(1)

    predict = str(y_hat.item())
    if predict=='0': predict = 'esophagitis'
    elif predict=='1': predict = 'polyp'
    else: predict = 'ulcerative colitis'

    probs = torch.nn.functional.softmax(outputs, dim=1)
    probs = probs.detach().numpy().flatten()
    probs = [(elem*100) for elem in probs]
    probs = ['%.3f' % elem for elem in probs]
    
    classes = ['Esophagitis', 'Polyp', 'Ulcerative colitis']
    probs = dict(zip(classes, probs))
    return predict, probs
    
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part.')
        return redirect(request.url)

    file_urls = [] # list to hold uploaded image urls
    files = request.files.getlist('files[]')

    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename.lower())) # save file to folder
            file_urls.append(UPLOAD_FOLDER + filename) # append image urls
        
        else:
            flash('Allowed image types are - png, jpg, and jpeg.')
            return redirect(request.url)
        
    return redirect(url_for('display', filename=file_urls))    

@app.route('/display/', methods=['GET','POST'])
def display():
    file_urls = request.args.getlist('filename')
    file_urls_with_path = ['../' + x for x in file_urls]
    total_img = len(file_urls)

    results = dict(zip(file_urls_with_path, repeat(None)))
    eso = polyp = ulcer = 0

    for i in range(total_img):
        temp_results = dict(zip(['Predict', 'Probs'], repeat(None)))
        predict, probs = get_prediction(file_urls[i])
        if predict == 'esophagitis': eso = eso+1
        elif predict == 'polyp': polyp = polyp+1
        else: ulcer = ulcer+1
        temp_results['Predict']= predict
        temp_results['Probs'] = probs
        results[file_urls_with_path[i]] = temp_results
    
    disease_count = dict(zip(['Esophagitis', 'Polyp', 'Ulcerative colitis'], [eso, polyp, ulcer]))

    return render_template('results.html', filename=file_urls_with_path, disease_count=disease_count, results=results)

if __name__ == "__main__":
    app.run()