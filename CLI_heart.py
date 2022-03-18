

import heart_attack_predictions as hap
model = hap.DecisionTreeClassifier
answer = []
while True:

    age = answer.append(int(input("How old are you? \n")))
    sex = answer.append(int(input("What sex are you? Male = 0 and female = 1 \n")))
    caa = answer.append(int(input("number of major vessels (0-3)? \n")))
    cp = answer.append(int(input("Chest Pain type chest pain type, between 1 and 4 \n")))
    trtbps = answer.append(int(input("resting blood pressure (in mm Hg) \n")))
    chol = answer.append(int(input("cholestoral in mg/dl fetched via BMI sensor \n")))
    fbs = answer.append(int(input("(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) \n")))
    restecg = answer.append(int(input("resting electrocardiographic results, between 0 and 2 \n")))
    thalachh = answer.append(int(input("maximum heart rate achieved \n")))
    exng = answer.append(int(input("exercise induced angina (1 = yes; 0 = no) \n")))
    oldpeak = answer.append(float(input("Oldpeak \n")))
    slp = answer.append(int(input("slp \n")))
    thall = answer.append(int(input("thall\n")))
    break

pred = model.predict(answer)
if pred == 0:
    print('You are healthy')
else:
    print('Get some medication')