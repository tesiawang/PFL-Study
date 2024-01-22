import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('recorder_cfg00.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc00 = recorder['mean_personalized_acc']
    mean_gen_acc00 = recorder['mean_generalized_acc']

with open('recorder_cfg01.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc01 = recorder['mean_personalized_acc']
    mean_gen_acc01 = recorder['mean_generalized_acc']

with open('recorder_cfg11.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc11 = recorder['mean_personalized_acc']
    mean_gen_acc11 = recorder['mean_generalized_acc']
# Plot
plt.figure(1)
plt.plot(mean_per_acc00, label='Personalized Accuracy (Simplest Alternating)')
plt.plot(mean_per_acc01, label='Personalized Accuracy (Alternating with only data quantity considered)')
plt.plot(mean_per_acc11, label='Personalized Accuracy (Alternating with weighted initial)')
plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Personalized Accuracy')


plt.figure(2)
plt.plot(mean_gen_acc00, label='Generalized Accuracy (Simplest Alternating)')
plt.plot(mean_gen_acc01, label='Generalized Accuracy (Alternating with only data quantity considered)')
plt.plot(mean_gen_acc11, label='Generalized Accuracy (Alternating with weighted initial)')
plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Generalized Accuracy')
plt.show()