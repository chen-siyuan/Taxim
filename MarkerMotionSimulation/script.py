#  import numpy as np
#  from matplotlib import pyplot as plt
#  import seaborn as sns

#  tensor_map = np.load(
        #  "../calibs/femCalib.npz",
        #  allow_pickle=True
        #  )["tensorMap"].transpose(1, 0, 2, 3)

#  for i in range(3):
    #  for j in range(3):
        #  plt.subplot(3, 3, i * 3 + j + 1)
        #  sns.heatmap(abs(tensor_map[:, :, i, j]) > np.max(tensor_map[:, :, i, j]) * 0.01)
        #  plt.axis("off")

#  plt.show()

#  gel_map = np.load("GM.npy").transpose(1, 0);
#  sns.heatmap(gel_map)
#  plt.savefig("heatmap")
