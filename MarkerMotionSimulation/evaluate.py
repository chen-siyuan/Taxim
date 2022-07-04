import numpy as np
from matplotlib import pyplot as plt


def compare_error(gtMap, composeMap, outfile):
    error_x = gtMap[1, :, :] - composeMap[1, :, :]
    error_y = gtMap[2, :, :] - composeMap[2, :, :]
    error_z = gtMap[3, :, :] - composeMap[3, :, :]
    avg_error_x = np.mean(np.abs(error_x))
    avg_error_y = np.mean(np.abs(error_y))
    avg_error_z = np.mean(np.abs(error_z))
    avg_error_xy = np.mean(np.sqrt(error_x ** 2 + error_y ** 2))
    print("error of dx is " + str("{:.3f}".format(np.mean(np.abs(error_x)))))
    print("error of dy is " + str("{:.3f}".format(np.mean(np.abs(error_y)))))
    print("error of dz is " + str("{:.3f}".format(np.mean(np.abs(error_z)))))
    print("error of dxdy is " + str(
        "{:.3f}".format(np.mean(np.sqrt(error_x ** 2 + error_y ** 2)))))
    outfile.write("error of dx is " + str(
        "{:.3f}".format(np.mean(np.abs(error_x)))) + '\n')
    outfile.write("error of dy is " + str(
        "{:.3f}".format(np.mean(np.abs(error_y)))) + '\n')
    outfile.write("error of dz is " + str(
        "{:.3f}".format(np.mean(np.abs(error_z)))) + '\n')
    outfile.write("error of dxdy is " + str(
        "{:.3f}".format(np.mean(np.sqrt(error_x ** 2 + error_y ** 2)))) + '\n')

    # relative_error_x = np.mean(np.abs(error_x/(gtMap[1,:,:]+1e-6)))
    # relative_error_y = np.mean(np.abs(error_y/(gtMap[2,:,:]+1e-6)))
    # relative_error_z = np.mean(np.abs(error_z/(gtMap[3,:,:]+1e-6)))
    relative_error_x = np.mean(
        np.abs(error_x / np.mean(np.abs(gtMap[1, :, :]))))
    relative_error_y = np.mean(
        np.abs(error_y / np.mean(np.abs(gtMap[2, :, :]))))
    relative_error_z = np.mean(
        np.abs(error_z / np.mean(np.abs(gtMap[3, :, :]))))
    relative_error_xy = np.mean(np.sqrt(error_x ** 2 + error_y ** 2) / (np.mean(np.abs(gtMap[1, :, :])) ** 2 + np.mean(np.abs(gtMap[1, :, :])) ** 2))
    print("relative error of dx is " + str("{:.3f}".format(
        np.mean(np.abs(error_x / np.mean(np.abs(gtMap[1, :, :])))))))
    print("relative error of dy is " + str("{:.3f}".format(
        np.mean(np.abs(error_y / np.mean(np.abs(gtMap[2, :, :])))))))
    print("relative error of dz is " + str("{:.3f}".format(
        np.mean(np.abs(error_z / np.mean(np.abs(gtMap[3, :, :])))))))
    print(
        "relative error of dxdy is " + str("{:.3f}".format(relative_error_xy)))
    outfile.write("relative error of dx is " + str("{:.3f}".format(
        np.mean(np.abs(error_x / np.mean(np.abs(gtMap[1, :, :])))))) + '\n')
    outfile.write("relative error of dy is " + str("{:.3f}".format(
        np.mean(np.abs(error_y / np.mean(np.abs(gtMap[2, :, :])))))) + '\n')
    outfile.write("relative error of dz is " + str("{:.3f}".format(
        np.mean(np.abs(error_z / np.mean(np.abs(gtMap[3, :, :])))))) + '\n')
    outfile.write("relative error of dxdy is " + str(
        "{:.3f}".format(relative_error_xy)) + '\n')

    plt.figure(1)
    plt.subplot(311)
    plt.imshow(error_x)

    plt.subplot(312)
    plt.imshow(error_y)

    plt.subplot(313)
    plt.imshow(error_z)
    plt.show()
