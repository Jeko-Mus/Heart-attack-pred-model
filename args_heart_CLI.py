import argparse
import numpy as np
from heart_attack_predictions import pred_heart
from lightgbm import LGBMClassifier
model = LGBMClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Heart attack predictor')

    parser.add_argument('age', type=int, help='enter your age')
    parser.add_argument('sex', type=int, choices=[0, 1] , help='Male = 1, Female = 0')
    parser.add_argument('cp', type=int, choices=[1, 2, 3, 4] , help='Chest Pain type , between 1 and 4')

    parser.add_argument('trtbps', type=int, help='resting blood pressure (in mm Hg)')
    parser.add_argument('chol', type=int, help='cholestoral in mg/dl fetched via BMI sensor')
    parser.add_argument('fbs', type=int, choices=[0, 1] , help='fasting blood sugar > 120 mg/dl) (1 = true; 0 = false')

    parser.add_argument('restecg', type=int, choices=[0, 1, 2] , help='resting electrocardiographic results, between 0 and 2')
    parser.add_argument('thalachh', type=int, help='maximum heart rate achieved')
    parser.add_argument('exng', type=int, choices=[0, 1] , help='exercise induced angina (1 = yes; 0 = no)')

    parser.add_argument('oldpeak', type=float, help='previous peak')
    parser.add_argument('slp', type=int, choices=[0, 1, 2] , help='slope')
    parser.add_argument('caa', type=int, choices=[0, 1, 2, 3] , help='number of major vessels (0-3)')
    parser.add_argument('thall', type=int, choices=[0, 1, 2, 3] , help='Thall rate')
    args = parser.parse_args()
    
    args_list = [args.age, args.sex, args.cp, args.trtbps, args.chol, args.fbs, args.restecg, args.thalachh, args.exng, args.oldpeak, 
                args.slp, args.caa, args.thall]
    args_list = np.array(args_list)
    args_list = args_list.reshape((1, -1))
    pred = pred_heart(args_list)
    print(pred)