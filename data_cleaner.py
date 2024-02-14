import pandas as pd
import numpy as np

open('metadata.txt', 'w+').close()
metadata = open('metadata.txt', 'a')


# def normalize(column: pd.Series):
#     mean = np.mean(column)
#     std = np.std(column)
#     metadata.write(f'{column.name} {mean} {std}\n')
#     return (column - mean)/std

def normalize(column: pd.Series):
    max = np.max(column)
    metadata.write(f'{column.name} {max}\n')
    return column / max


df = pd.read_excel('heart_attack_prediction_dataset.xlsx')
df = df.drop('Patient ID', axis=1)
df = df.drop('Country', axis=1)
df = df.drop('Continent', axis=1)
df = df.drop('Hemisphere', axis=1)

df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split(
    '/', expand=True).astype('float64')
df = df.drop('Blood Pressure', axis=1)
column_to_move = df.pop('Heart Attack Risk')
df.insert(len(df.columns), 'Heart Attack Risk', column_to_move)

# df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})
# df['Diet'] = df['Diet'].replace({'Unhealthy': 0, 'Average': 0.5, 'Healthy': 1})

df['Age'] = normalize(df['Age'])
df['Cholesterol'] = normalize(df['Cholesterol'])
# df['Heart Rate'] = normalize(df['Heart Rate'])
# df['Exercise Hours Per Week'] = normalize(df['Exercise Hours Per Week'])
# df['Stress Level'] = df['Stress Level'] / 10
# df['Sedentary Hours Per Day'] = normalize(df['Sedentary Hours Per Day'])
# df['Income'] = normalize(df['Income'])
df['BMI'] = normalize(df['BMI'])
df['Triglycerides'] = normalize(df['Triglycerides'])
df['Physical Activity Days Per Week'] = normalize(df['Physical Activity Days Per Week'])
# df['Sleep Hours Per Day'] = normalize(df['Sleep Hours Per Day'])
df['Systolic'] = normalize(df['Systolic'])
df['Diastolic'] = normalize(df['Diastolic'])

df = df.drop('Sex', axis=1)
df = df.drop('Diet', axis=1)
# df = df.drop('Smoking', axis=1)
# df = df.drop('Diabetes', axis=1)
df = df.drop('Alcohol Consumption', axis=1)
# df = df.drop('Age', axis=1)
df = df.drop('Heart Rate', axis=1)
df = df.drop('Exercise Hours Per Week', axis=1)
df = df.drop('Stress Level', axis=1)
df = df.drop('Sedentary Hours Per Day', axis=1)
df = df.drop('Income', axis=1)
# df = df.drop('BMI', axis=1)
# df = df.drop('Physical Activity Days Per Week', axis=1)
df = df.drop('Sleep Hours Per Day', axis=1)
df = df.drop('Family History', axis=1)
df = df.drop('Obesity', axis=1)
df = df.drop('Previous Heart Problems', axis=1)
df = df.drop('Medication Use', axis=1)
df = df.drop('Triglycerides', axis=1)

# print(df['Heart Attack Risk'].value_counts()[1])  # 3139
# print(df['Heart Attack Risk'].value_counts()[0])  # 5624
amount = 5624 - 3139
# while amount:
row_to_remove = np.random.choice(
    df.loc[df['Heart Attack Risk'] == 0].index, amount, replace=False)
df = df.drop(row_to_remove)
df = df.sample(frac=1)

print(f'{" DATA ":#^100}')
print(df)
df.to_csv('data.csv', index=False)
metadata.close()
