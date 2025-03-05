# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask


# load packages==============================================================
from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the scaler, label encoder, model, and class names=====================
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
class_names = [ 'B.Arch- Bachelor of Architecture',
     'B.Com- Bachelor of Commerce',
     'B.Ed.',
     'B.Sc- Applied Geology',
     'B.Sc- Nursing',
     'B.Sc. Chemistry',
     'B.Sc. Mathematics',
     'B.Sc.- Information Technology',
     'B.Sc.- Physics',
     'B.Tech.-Civil Engineering',
     'B.Tech.-Computer Science and Engineering',
     'B.Tech.-Electrical and Electronics Engineering',
     'B.Tech.-Electronics and Communication Engineering',
     'B.Tech.-Mechanical Engineering',
     'BA in Economics',
     'BA in English',
     'BA in Hindi',
     'BA in History',
     'BBA- Bachelor of Business Administration',
     'BBS- Bachelor of Business Studies',
     'BCA- Bachelor of Computer Applications',
     'BDS- Bachelor of Dental Surgery',
     'BEM- Bachelor of Event Management',
     'BFD- Bachelor of Fashion Designing',
     'BJMC- Bachelor of Journalism and Mass Communication',
     'BPharma- Bachelor of Pharmacy',
     'BTTM- Bachelor of Travel and Tourism Management',
     'BVA- Bachelor of Visual Arts',
     'CA- Chartered Accountancy',
     'CS- Company Secretary',
     'Civil Services',
     'Diploma in Dramatic Arts',
     'Integrated Law Course- BA + LL.B',
     'MBBS']

# Recommendations ===========================================================
def Recommendations(Drawing, Dancing, Singing, Sports, Video_Game, Acting, Travelling, Gardening, Animals, Photography, Teaching, Exercise, Coding, Electricity_Components, Mechanic_Parts, Computer_Parts, Researching, Architecture, Historic_Collection, Botany, Zoology, Physics, Accounting, Economics, Sociology, Geography, Psychology, History, Science, Business_Education, Chemistry, Mathematics, Biology,  Designing, Content_Writing, Crafting, Literature, Reading, Cartooning, Debating, Astrology, Hindi, French, English, Solving_Puzzles, Gymnastics, Yoga, Engineering, Doctor, Pharmacist, Cycling, Knitting, Director, Journalism, Business, Listening_to_Music):
    
    # Create feature array
    feature_array = np.array([[
        Drawing, Dancing, Singing, Sports, Video_Game, Acting, Travelling, Gardening, 
        Animals, Photography, Teaching, Exercise, Coding, Electricity_Components, 
        Mechanic_Parts, Computer_Parts, Researching, Architecture, Historic_Collection, 
        Botany, Zoology, Physics, Accounting, Economics, Sociology, Geography, Psychology, 
        History, Science, Business_Education, Chemistry, Mathematics, Biology,  
        Designing, Content_Writing, Crafting, Literature, Reading, Cartooning, Debating, 
        Astrology, Hindi, French, English, Solving_Puzzles, Gymnastics, Yoga, Engineering, 
        Doctor, Pharmacist, Cycling, Knitting, Director, Journalism, Business, Listening_to_Music
    ]])
   # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict using the model
    probabilities = model.predict_proba(scaled_features)
    # Get top five predicted classes along with their probabilities
   
    top_classes_idx = np.argsort(-probabilities[0])[:10]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST', 'GET'])



def pred():
    if request.method == 'POST':
        # Extract user inputs from form
        user_inputs = {key: int(request.form[key]) for key in request.form}
        
        # Get recommendations from the model
        recommendations = Recommendations(**user_inputs)
        
        # Extract course names and probabilities
        courses = [rec[0] for rec in recommendations]
        probabilities = [rec[1] for rec in recommendations]

        # Create Bar Chart
        plt.figure(figsize=(10, 6))
        plt.bar(courses, probabilities, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.xlabel("Courses")
        plt.ylabel("Probability")
        plt.title("Probability Distribution of Career Paths")

        # Add probability labels on bars
        for i, v in enumerate(probabilities):
            plt.text(i, v + 0.004, str(round(v, 2)), ha='center', fontsize=12)

        # Save the graph
        img_path = os.path.join("static", "recommendation_chart.png") 
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better visibility
        plt.savefig(img_path)
        plt.close()

        return render_template('results.html', recommendations=recommendations, img_path=img_path)
    
    return render_template('home.html')

    if request.method == 'POST':
        Drawing = int(request.form['Drawing'])
        Dancing = int(request.form['Dancing'])
        Singing = int(request.form['Singing'])
        Sports = int(request.form['Sports'])
        Video_Game = int(request.form['Video_Game'])
        Acting = int(request.form['Acting'])
        Travelling = int(request.form['Travelling'])
        Gardening = int(request.form['Gardening'])
        Animals = int(request.form['Animals'])
        Photography = int(request.form['Photography'])
        Teaching = int(request.form['Teaching'])
        Exercise = int(request.form['Exercise'])
        Coding = int(request.form['Coding'])
        Electricity_Components = int(request.form['Electricity_Components'])
        Mechanic_Parts = int(request.form['Mechanic_Parts'])
        Computer_Parts = int(request.form['Computer_Parts'])
        Researching = int(request.form['Researching'])
        Architecture = int(request.form['Architecture'])
        Historic_Collection = int(request.form['Historic_Collection'])
        Botany = int(request.form['Botany'])
        Zoology = int(request.form['Zoology'])
        Physics = int(request.form['Physics'])
        Accounting = int(request.form['Accounting'])
        Economics = int(request.form['Economics'])
        Sociology = int(request.form['Sociology'])
        Geography = int(request.form['Geography'])
        Psychology = int(request.form['Psychology'])
        History = int(request.form['History'])
        Science = int(request.form['Science'])
        Business_Education = int(request.form['Business_Education'])
        Chemistry = int(request.form['Chemistry'])
        Mathematics = int(request.form['Mathematics'])
        Biology = int(request.form['Biology'])
       
        Designing = int(request.form['Designing'])
        Content_Writing = int(request.form['Content_Writing'])
        Crafting = int(request.form['Crafting'])
        Literature = int(request.form['Literature'])
        Reading = int(request.form['Reading'])
        Cartooning = int(request.form['Cartooning'])
        Debating = int(request.form['Debating'])
        Astrology = int(request.form['Astrology'])
        Hindi = int(request.form['Hindi'])
        French = int(request.form['French'])
        English = int(request.form['English'])
        Solving_Puzzles = int(request.form['Solving_Puzzles'])
        Gymnastics = int(request.form['Gymnastics'])
        Yoga = int(request.form['Yoga'])
        Engineering = int(request.form['Engineering'])
        Doctor = int(request.form['Doctor'])
        Pharmacist = int(request.form['Pharmacist'])
        Cycling = int(request.form['Cycling'])
        Knitting = int(request.form['Knitting'])
        Director = int(request.form['Director'])
        Journalism = int(request.form['Journalism'])
        Business = int(request.form['Business'])
        Listening_to_Music = int(request.form['Listening_to_Music'])


        recommendations = Recommendations(Drawing, Dancing, Singing, Sports, Video_Game, Acting, Travelling, Gardening, Animals, Photography, Teaching, Exercise, Coding, Electricity_Components, Mechanic_Parts, Computer_Parts, Researching, Architecture, Historic_Collection, Botany, Zoology, Physics, Accounting, Economics, Sociology, Geography, Psychology, History, Science, Business_Education, Chemistry, Mathematics, Biology, Designing, Content_Writing, Crafting, Literature, Reading, Cartooning, Debating, Astrology, Hindi, French, English, Solving_Puzzles, Gymnastics, Yoga, Engineering, Doctor, Pharmacist, Cycling, Knitting, Director, Journalism, Business, Listening_to_Music)

        return render_template('results.html', recommendations=recommendations)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)