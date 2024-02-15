import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('recorder_weighted_new_obj0.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc0 = recorder['mean_personalized_acc']
    mean_gen_acc0 = recorder['mean_generalized_acc']

# with open('recorder_new_obj1.pkl', 'rb') as f:
#     recorder = pickle.load(f)
#     mean_per_acc1 = recorder['mean_personalized_acc']
#     mean_gen_acc1 = recorder['mean_generalized_acc']
#
# with open('recorder_new_obj2.pkl', 'rb') as f:
#     recorder = pickle.load(f)
#     mean_per_acc2 = recorder['mean_personalized_acc']
#     mean_gen_acc2 = recorder['mean_generalized_acc']

with open('recorder_weighted_new_obj4.pkl', 'rb') as f:
    recorder = pickle.load(f)
    mean_per_acc4 = recorder['mean_personalized_acc']
    mean_gen_acc4 = recorder['mean_generalized_acc']

# print(mean_per_acc4[65:69])
# print(mean_per_acc0[55:59])
# with open('recorder_weighted_new_obj2.pkl', 'rb') as f:
#     recorder = pickle.load(f)
#     mean_per_acc = recorder['mean_personalized_acc']
#     mean_gen_acc = recorder['mean_generalized_acc']

# Plot
plt.figure(1)
plt.plot(np.array(mean_per_acc0[0:59]), label='Personalized Accuracy (existing objective)')
# plt.plot(mean_per_acc1, label='Personalized Accuracy (only the first order)')
# plt.plot(mean_per_acc2, label='Personalized Accuracy (new loss)')
plt.plot(mean_per_acc4[0:59], label='Personalized Accuracy (new error bound)')
# plt.plot(mean_per_acc, label='Personalized Accuracy (new loss with weighted initial)')

plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Personalized Accuracy')


plt.figure(2)
plt.plot(mean_gen_acc0[0:59], label='Generalized Accuracy (existing objective)')
# plt.plot(mean_gen_acc1, label='Generalized Accuracy (only the first order)')
# plt.plot(mean_gen_acc2, label='Generalized Accuracy (new loss)')
plt.plot(mean_gen_acc4[0:59], label='Generalized Accuracy (new error bound)')
# plt.plot(mean_gen_acc, label='Generalized Accuracy (new loss with weighted initial)')
plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Generalized Accuracy')


# Generate a range of x values
# x = np.linspace(0, 10, 400)
#
# # Compute the corresponding y values
# y = np.sqrt(x)
#
# # Create the plot
# plt.figure(3)
# plt.plot(x, y)
plt.show()