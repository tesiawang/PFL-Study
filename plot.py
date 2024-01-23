import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('recorder_new_obj0.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc0 = recorder['mean_personalized_acc']
    mean_gen_acc0 = recorder['mean_generalized_acc']

with open('recorder_new_obj1.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc1 = recorder['mean_personalized_acc']
    mean_gen_acc1 = recorder['mean_generalized_acc']

with open('recorder_new_obj2.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc2 = recorder['mean_personalized_acc']
    mean_gen_acc2 = recorder['mean_generalized_acc']

with open('recorder_new_obj3.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc3 = recorder['mean_personalized_acc']
    mean_gen_acc3 = recorder['mean_generalized_acc']

# Plot
plt.figure(1)
plt.plot(mean_per_acc0, label='Personalized Accuracy (oringinal objective)')
plt.plot(mean_per_acc1, label='Personalized Accuracy (only the first order)')
plt.plot(mean_per_acc2, label='Personalized Accuracy (new loss)')
plt.plot(mean_per_acc3, label='Personalized Accuracy (gen error bound)')
plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Personalized Accuracy')


plt.figure(2)
plt.plot(mean_gen_acc0, label='Personalized Accuracy (oringinal objective)')
plt.plot(mean_gen_acc1, label='Personalized Accuracy (only the first order)')
plt.plot(mean_gen_acc2, label='Personalized Accuracy (new loss)')
plt.plot(mean_gen_acc3, label='Personalized Accuracy (gen error bound)')
plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Generalized Accuracy')
plt.show()